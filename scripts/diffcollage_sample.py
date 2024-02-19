"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

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
import diff_collage as dc


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
        # patchify=args.patchify,
        # half=True,
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

    overlap_size = 64  # how much overlap
    num_img = 3  # how many square images
    img_shape = (4, 16, 128)  # image shape    # todo: add in args

    def eps_fn(x, t, y=None, half=False):
        return model(x.permute(0, 1, 3, 2), t, y=y, half=half).permute(0, 1, 3, 2)  # since our backbone takes 128x16 as input

    worker = dc.CondIndCircle(img_shape, eps_fn, num_img+1, overlap_size=overlap_size)   # circle need one more num_img than linear
    # worker = dc.CondIndSimple(img_shape, eps_fn, num_img, overlap_size=overlap_size)
    model_long_fn = worker.eps_scalar_t_fn

    def model_fn(x, t, y=None):
        # takes in 4 x pitch x time, return 4 x pitch x time
        y_null = th.tensor([args.num_classes] * args.batch_size, device=dist_util.dev())
        if args.class_cond:
            return (1 + args.w) * model_long_fn(x, t, y) - args.w * model_long_fn(x, t, y_null)
        else:
            return model_long_fn(x, t, y_null)

    # create embed model
    embed_model = load_model(args.embed_model_name, args.embed_model_ckpt)
    embed_model.to(dist_util.dev())
    embed_model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    save_dir = os.path.join(logger.get_dir(), "generated_samples")
    os.makedirs(os.path.expanduser(save_dir), exist_ok=True)
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            # only generate one class
            classes = th.ones(size=(args.batch_size,), device=dist_util.dev(), dtype=th.int) * 0
            # randomly select classes
            # classes = th.randint(
            #     low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            # )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, *worker.shape),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            progress=True
        )
        sample = midi_util.decode_sample_for_midi(sample, embed_model=embed_model,
                                                  scale_factor=args.scale_factor, threshold=-0.95)

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

    arr = np.concatenate(all_images, axis=0)
    if arr.shape[-1] == 1:  # no pedal, need shape B x 128 x 1024
        arr = arr.squeeze(axis=-1)
    else:  # with pedal, need shape: B x 2 x 128 x 1024
        arr = arr.transpose(0, 3, 1, 2)
    arr = arr[: args.num_samples]
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
        w=4.,   # for cfg
        training=False,
        port=None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
