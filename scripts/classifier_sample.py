"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, midi_util, logger
from guided_diffusion.dit import DiT_models
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_diffusion,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from load_utils import load_model
from guided_diffusion.pr_datasets import FUNC_DICT


def main():
    args = create_argparser().parse_args()

    comm = dist_util.setup_dist(port=args.port)
    logger.configure(args=args, comm=comm)

    logger.log("creating model and diffusion...")
    model = DiT_models[args.model](
        input_size=args.image_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        learn_sigma=args.learn_sigma,
        patchify=args.patchify,
    )
    diffusion = create_diffusion(
        learn_sigma=args.learn_sigma,
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        timestep_respacing=args.timestep_respacing,
        use_kl=args.use_kl,
        predict_xstart=args.predict_xstart,
        rescale_timesteps=args.rescale_timesteps,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu"), strict=False
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = DiT_models[args.classifier_model](
        input_size=args.image_size,    # classifier trained on latents for now, so has the same img size as diffusion
        in_channels=args.in_channels,
        num_classes=args.classifier_num_classes,
        patchify=args.classifier_patchify,
    )
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    # y is a dummy input for cond_fn, rule is the real input
    def cond_fn_xentropy(x, t, y=None, rule=None):
        # Xentropy cond_fn
        assert rule is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), rule.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def cond_fn_mse(x, t, y=None, rule=None, var=0.1):
        # mse cond_fn, preset var
        assert rule is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = -1/(2*var) * F.mse_loss(logits, rule, reduction="none").sum(dim=-1)
            return th.autograd.grad(log_probs.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None, rule=None):
        # y has to be composer, rule is a dummy input
        y_null = th.tensor([args.num_classes] * args.batch_size, device=dist_util.dev())
        if args.class_cond:
            if args.cfg:
                return (1 + args.w) * model(x, t, y) - args.w * model(x, t, y_null)
            else:
                return model(x, t, y)
        else:
            return model(x, t, y_null)

    # create embed model
    embed_model = load_model(args.embed_model_name, args.embed_model_ckpt)
    embed_model.to(dist_util.dev())
    embed_model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            # only generate one class
            classes = th.ones(size=(args.batch_size,), device=dist_util.dev(), dtype=th.int) * 1
            # randomly select classes
            # classes = th.randint(
            #     low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            # )
            model_kwargs["y"] = classes
        if args.rule is not None:
            if args.rule == 'pitch_hist':
                # rule_label = th.zeros(size=(args.batch_size, 12), device=dist_util.dev())
                major_profile = th.tensor(
                    # [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
                    [6.35, 0., 0., 0., 4.38, 0., 0., 5.19, 0., 0., 0., 0.],
                    device=dist_util.dev())
                minor_profile = th.tensor(
                    # [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],   # C minor
                    # [3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34],    # C# minor
                    [6.33, 0., 0., 5.38, 0., 0., 0., 4.75, 0., 0., 0., 0.],
                    device=dist_util.dev())
                # rule_label_c_maj = major_profile.repeat(args.batch_size//2, 1)
                # rule_label_c_min = minor_profile.repeat(args.batch_size//2, 1)
                # rule_label = th.concat((rule_label_c_maj, rule_label_c_min), dim=0)
                if args.major:
                    rule_label = major_profile.repeat(args.batch_size, 1)
                else:
                    rule_label = minor_profile.repeat(args.batch_size, 1)
                rule_label = rule_label / (th.sum(rule_label, dim=-1, keepdim=True) + 1e-12)
            else:
                rule_label = None
            model_kwargs["rule"] = rule_label
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        if args.rule in ['']:   # leave for other category based rules
            cond_fn_used = cond_fn_xentropy
        elif args.rule in ['pitch_hist', 'note_density']:
            cond_fn_used = cond_fn_mse
        else:
            cond_fn_used = None

        sample = sample_fn(
            model_fn,
            (args.batch_size, args.in_channels, args.image_size[0], args.image_size[1]),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            cond_fn=cond_fn_used,
            progress=True
        )
        sample = sample / args.scale_factor
        image_size_h = args.image_size[0]
        image_size_w = args.image_size[1]

        if image_size_h != image_size_w:  # transposed for raster col
            sample = sample.permute(0, 1, 3, 2)  # vertical axis means pitch after transpose

        save_dir = os.path.join(logger.get_dir(), "generated_samples")
        os.makedirs(os.path.expanduser(save_dir), exist_ok=True)

        with th.no_grad():
            if embed_model is not None:
                if image_size_h != image_size_w:
                    sample = th.chunk(sample, image_size_h // image_size_w, dim=-1)  # B x C x H x W
                    sample = th.concat(sample, dim=0)  # 1st second for all batch, 2nd second for all batch, ...
                    # th.save(sample, save_dir + '/latents.pt')    # todo: uncomment if wants to save sampled latents
                if args.get_KL:
                    sample = embed_model.decode(sample)
                else:
                    sample = embed_model.decode_diff(sample)
                if image_size_h != image_size_w:
                    sample = th.concat(th.chunk(sample, image_size_h // image_size_w, dim=0), dim=-1)

        sample[sample <= -0.95] = -1.  # heuristic thresholding the background
        sample = ((sample + 1) * 63.5).clamp(0, 127).to(th.uint8)
        # # todo: comment above two lines to test no threshold
        # sample[sample <= -1.] = -1.
        # sample = (sample + 1) * 63.5

        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0).squeeze()
    arr = arr[: args.num_samples]
    # test distance between generated samples and target
    generated_samples = th.from_numpy(arr) / 63.5 - 1
    loss = []
    rule_label = rule_label.cpu()
    for i in range(generated_samples.shape[0]):
        gen_rule = FUNC_DICT[args.rule](generated_samples[i][None][None])
        loss.append(F.mse_loss(rule_label, gen_rule).item())
    mean_loss = th.tensor(loss).mean()
    print(f"loss: {mean_loss}")    # distance between generated rule and target rule

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        if args.class_cond:
            midi_util.save_piano_roll_midi(arr, save_dir, args.fs, y=label_arr)
        else:
            midi_util.save_piano_roll_midi(arr, save_dir, args.fs)
    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        project="music-sampling",
        dir="",
        model="DiT-B/4",  # DiT model names
        embed_model_name='kl/f8',
        embed_model_ckpt='kl-f8-4-6/steps_42k.ckpt',
        classifier_use_fp16=False,
        classifier_model='DiTRel-XS/2-cls',
        classifier_num_classes=12,
        classifier_patchify=True,
        classifier_path='loggings/classifier-pitch-hist/ditrel-noise/model004999.pt',
        clip_denoised=False,
        num_samples=128,
        batch_size=16,
        use_ddim=False,
        model_path="",
        get_KL=True,
        scale_factor=1.,
        patchify=True,
        fs=100,
        num_classes=0,
        cfg=False,
        w=4.,   # for cfg
        classifier_scale=1.0,
        rule=None,
        major=True,
        port=None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
