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


def main():
    args = create_argparser().parse_args()

    comm = dist_util.setup_dist(port=args.port)
    logger.configure(args=args, comm=comm)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu"), strict=False
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    def model_fn(x, t, y=None):
        y_null = th.tensor([args.num_classes] * args.batch_size, device=dist_util.dev())
        if args.class_cond:
            if args.cfg:
                # the original unet architecture does not support cfg, always using w=0
                return (1 + args.w) * model(x, t, y) - args.w * model(x, t, y_null)
            else:
                return model(x, t, y)
        else:
            return model(x, t, y_null)

    # create embed model
    embed_model = None

    logger.log("sampling...")
    all_images = []
    all_labels = []
    save_dir = os.path.join(logger.get_dir(), f"gen_cls_{args.class_label}{args.save_name}")
    os.makedirs(os.path.expanduser(save_dir), exist_ok=True)
    print(save_dir)
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            # only generate one class
            classes = th.ones(size=(args.batch_size,), device=dist_util.dev(), dtype=th.int) * args.class_label
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
            (args.batch_size, args.in_channels, args.image_size[0], args.image_size[1]),
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
        save_name="",  # optional save name to indicate iter
        dir="",
        model="",  # DiT model names
        clip_denoised=False,
        num_samples=128,
        batch_size=16,
        use_ddim=False,
        model_path="",
        scale_factor=1.,
        fs=12.5,
        num_classes=3,
        class_label=1,  # class to generate
        cfg=False,
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
