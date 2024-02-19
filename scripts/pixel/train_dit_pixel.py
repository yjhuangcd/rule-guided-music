"""
Train a diffusion model on images.
"""

import argparse
import os
import os.path as osp

from guided_diffusion import dist_util, logger
from guided_diffusion.dit import DiT_models
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.pr_datasets_all import load_data
from load_utils import load_model
from music_score_sde.models import utils as mutils
from music_score_sde.utils import restore_checkpoint
from mpi4py import MPI
from absl import app
from absl import flags
from absl.flags import argparse_flags


def main(args):
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
    model.to(dist_util.dev())
    # create model architecture for eval loss, need to use ema params
    eval_model = DiT_models[args.model](
        input_size=args.image_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        learn_sigma=args.learn_sigma,
        patchify=args.patchify,
    )
    eval_model.to(dist_util.dev())
    # create embed model
    if args.embed_model_name is not None:
        embed_model = load_model(args.embed_model_name, args.embed_model_ckpt)
        del embed_model.loss
        embed_model.to(dist_util.dev())
        embed_model.eval()

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir + "_train.csv",
        batch_size=args.batch_size // args.encode_rep,
        class_cond=args.class_cond,
        image_size=args.pr_image_size,
    )

    eval_data = load_data(
        data_dir=args.data_dir + "_test.csv",
        batch_size=args.batch_size // args.encode_rep,
        class_cond=args.class_cond,
        image_size=args.pr_image_size,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        eval_model=eval_model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        embed_model=embed_model if args.embed_model_name is not None else None,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        eval_data=eval_data,
        eval_interval=args.save_interval,
        eval_sample_batch_size=16,
        total_num_gpus = MPI.COMM_WORLD.Get_size(),
        eval_sample_use_ddim=False,
        eval_sample_clip_denoised=args.eval_sample_clip_denoised,    # do not clip when training on latent space
        in_channels=args.in_channels,
        fs=args.fs,
        pedal=args.pedal,
        scale_factor=args.scale_factor,   # need to manually set scale_factor when resume
        num_classes=args.num_classes,   # whether to use class_cond in sampling
        microbatch_encode=args.microbatch_encode,
        encode_rep=args.encode_rep,
        shift_size=args.shift_size,
    ).run_loop()


def parse_flags(argv):
    parser = argparse_flags.ArgumentParser(description='An argparse + app.run example')
    defaults = dict(
        project="music-guided",
        dir="",
        data_dir="",
        model="DiT-XL/8",  # DiT model names
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,   # total steps, if set to be a positive number, lr will linearly decay
        batch_size=1,
        encode_rep=1,  # fixed to be 1 for pixel space
        shift_size=2,  # dummy for pixel space
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        embed_model_name=None,
        embed_model_ckpt=None,
        eval_sample_clip_denoised=False,
        discriminator_eval=False,
        scale_factor=1.,
        patchify=True,
        fs=12.5,   # saving piano roll with this fs
        pedal=False,
        num_classes=0,    # 0 is unconditional
        microbatch_encode=-1,
        pr_image_size=128,
        training=True,
        port=None,   # whether to use dist setup on ngc
    )
    defaults.update(model_and_diffusion_defaults())
    add_dict_to_argparser(parser, defaults)
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    app.run(main, flags_parser=parse_flags)
