"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os
import os.path as osp

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.pr_datasets_all import load_data
from guided_diffusion.dit import DiT_models
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from load_utils import load_model
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict, get_kl_input


def main():
    args = create_argparser().parse_args()

    comm = dist_util.setup_dist(port=args.port)
    logger.configure(args=args, comm=comm)

    logger.log("creating model and diffusion...")
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    # create embed model
    if args.embed_model_name is not None:
        embed_model = load_model(args.embed_model_name, args.embed_model_ckpt)
        del embed_model.loss
        embed_model.to(dist_util.dev())
        embed_model.eval()

    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            model.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint, map_location=dist_util.dev()
                )
            )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir + "_train.csv",
        batch_size=args.batch_size // args.encode_rep,
        class_cond=True if args.rule is None else False,
        image_size=args.pr_image_size,
        rule=args.rule,
    )
    if args.val_data_dir:
        val_data = load_data(
            data_dir=args.data_dir + "_test.csv",
            batch_size=args.batch_size // args.encode_rep,
            class_cond=True if args.rule is None else False,
            image_size=args.pr_image_size,
            rule=args.rule,
        )
    else:
        val_data = None

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("training classifier model...")

    def forward_backward_log(data_loader, prefix="train", rule=None):
        batch, extra = next(data_loader)
        if rule is not None:
            if rule == 'chord_progression_pixel':
                labels_key = extra["key"].to(dist_util.dev())   # B x 1
                labels_chord = extra["chord"].to(dist_util.dev())   # B x 8
                labels = th.concat((labels_key, labels_chord), dim=-1)   # B x (1+8)
            else:
                labels = extra[rule].to(dist_util.dev())
        else:
            labels = extra["y"].to(dist_util.dev())
        if args.get_KL:
            # need more sample diversity in a batch for classification
            batch = get_kl_input(batch, microbatch=args.microbatch_encode, model=embed_model,
                                 scale_factor=args.scale_factor, recombine=False)
        if batch.shape[0] != labels.shape[0]:
            labels = labels.repeat_interleave(args.encode_rep, dim=0)
        batch = batch.to(dist_util.dev())
        # Noisy images
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            # decoder cannot decode samples with t < 750
            if args.no_high_noise:
                t[t > 750] = 1000 - t[t > 750]
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
            if rule == 'chord_progression_pixel':
                key, chord = model(sub_batch, sub_t)
            else:
                logits = model(sub_batch, sub_t)
            if rule is not None:
                if rule == 'chord_progression_pixel':
                    sub_labels_key = sub_labels[:, :1].squeeze()
                    sub_labels_chord = sub_labels[:, 1:].reshape(-1)
                    chord = chord.reshape(-1, chord.shape[-1])
                    loss_key = F.cross_entropy(key, sub_labels_key, reduction="none")
                    loss_chord = F.cross_entropy(chord, sub_labels_chord, reduction="none")
                    # reshape to B x n_chord (8), and average along n_chord
                    loss_chord = loss_chord.reshape(sub_batch.shape[0], -1).mean(dim=-1)
                    loss = (loss_key + loss_chord) / 2
                else:
                    loss = F.mse_loss(logits, sub_labels, reduction="none").mean(dim=-1)
            else:   # train for cfg condition
                loss = F.cross_entropy(logits, sub_labels, reduction="none")

            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            if rule is None:
                losses[f"{prefix}_acc@1"] = compute_top_k(
                    logits, sub_labels, k=1, reduction="none"
                )
                # losses[f"{prefix}_acc@5"] = compute_top_k(
                #     logits, sub_labels, k=5, reduction="none"
                # )
            elif rule == 'chord_progression_pixel':
                losses[f"{prefix}_acc@1"] = compute_top_k(
                    chord, sub_labels_chord, k=1, reduction="none"
                )
            log_loss_dict(diffusion, sub_t, losses)
            del losses
            loss = loss.mean()
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        forward_backward_log(data, rule=args.rule)
        mp_trainer.optimize(opt)
        if val_data is not None and not step % args.eval_interval:
            with th.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(val_data, prefix="val", rule=args.rule)
                    model.train()
        if not step % args.log_interval:
            logger.dumpkvs()
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        project="music-guided-classifier",
        dir="",
        data_dir="",
        val_data_dir="",
        in_channels=3,
        noised=True,
        no_high_noise=False,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        encode_rep=1,   # whether to use recombination of encoded excerpts
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=5,
        save_interval=10000,
        get_KL=False,
        scale_factor=1.,
        embed_model_name=None,
        embed_model_ckpt=None,
        microbatch_encode=-1,
        pr_image_size=128,
        rule=None,
        chord=False,  # set to true if train for chord
        num_classes=9,  # number of outputs from classifier
        training=False,   # not training diffusion
        port=None,     # whether to use fixed port for ngc
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
