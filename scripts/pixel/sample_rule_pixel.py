"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import pandas as pd
import torch.distributed as dist
import torch.nn.functional as F
import multiprocessing

from guided_diffusion import dist_util, midi_util, logger
from guided_diffusion.dit import DiT_models
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    create_classifier,
)
from guided_diffusion.gaussian_diffusion import _encode, _extract_rule
from guided_diffusion.pr_datasets_all import load_data
from load_utils import load_model
import diff_collage as dc
from guided_diffusion.condition_functions import (
    model_fn, dc_model_fn, composite_nn_zt, composite_rule)
from functools import partial
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def main():
    args = create_argparser().parse_args()
    root_dir = 'cond_table/'
    args.dir = root_dir + os.path.splitext(args.config_path.split(root_dir)[-1])[0] + f'_cls_{args.class_label}'

    comm = dist_util.setup_dist(port=args.port)
    logger.configure(args=args, comm=comm)
    config = midi_util.load_config(args.config_path)
    if config.sampling.use_ddim:
        args.timestep_respacing = config.sampling.timestep_respacing

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

    # create embed model
    embed_model = None

    cond_fn_config = config.guidance.cond_fn
    if config.guidance.nn:
        logger.log("loading classifier...")
        classifier_config = cond_fn_config.classifiers
        num_classifiers = len(classifier_config.paths)
        classifiers = []
        for i in range(num_classifiers):
            classifier = create_classifier(
                image_size=args.image_size,
                in_channels=args.in_channels,
                classifier_use_fp16=False,
                classifier_width=128,
                classifier_depth=2,
                classifier_attention_resolutions='32,16,8',
                classifier_use_scale_shift_norm=True,
                classifier_resblock_updown=True,
                classifier_pool='attention',
                num_classes=classifier_config.num_classes[i],
                chord=True if 'chord' in cond_fn_config.rule_names[i] else False,
            )
            classifier.load_state_dict(
                dist_util.load_state_dict(classifier_config.paths[i], map_location="cpu")
            )
            classifier.to(dist_util.dev())
            classifier.eval()
            classifiers.append(classifier)

    if cond_fn_config is not None:
        if config.guidance.nn:
            cond_fn_used = partial(composite_nn_zt, fns=cond_fn_config.fns,
                                   classifier_scales=cond_fn_config.classifier_scales,
                                   classifiers=classifiers, rule_names=cond_fn_config.rule_names)
        else:
            cond_fn_used = partial(composite_rule, fns=cond_fn_config.fns,
                                   classifier_scales=cond_fn_config.classifier_scales,
                                   rule_names=cond_fn_config.rule_names)
    else:
        cond_fn_used = None

    if config.sampling.diff_collage:
        def eps_fn(x, t, y=None, half=False):
            # since our backbone takes 128x16 as input
            return model(x.permute(0, 1, 3, 2), t, y=y, half=half).permute(0, 1, 3, 2)

        # circle need one more num_img than linear
        img_shape = (args.in_channels, args.image_size[1], args.image_size[0])  # 4 x 16 x 128
        if config.dc.type == 'circle':
            worker = dc.CondIndCircle(img_shape, eps_fn, config.dc.num_img + 1, overlap_size=config.dc.overlap_size)
        else:
            worker = dc.CondIndSimple(img_shape, eps_fn, config.dc.num_img, overlap_size=config.dc.overlap_size)
        model_long_fn = worker.eps_scalar_t_fn
        gen_shape = (args.batch_size, worker.shape[0], worker.shape[2], worker.shape[1])
        model_fn_used = partial(dc_model_fn, model=model_long_fn, num_classes=args.num_classes,
                                class_cond=args.class_cond, cfg=args.cfg, w=args.w)
    else:
        gen_shape = (args.batch_size, args.in_channels, args.image_size[0], args.image_size[1])
        model_fn_used = partial(model_fn, model=model, num_classes=args.num_classes,
                                class_cond=args.class_cond, cfg=args.cfg, w=args.w)

    if args.class_cond:
        # only generate one class
        classes = th.ones(size=(args.batch_size,), device=dist_util.dev(), dtype=th.int) * args.class_label

    save_dir = logger.get_dir()
    save_dir_gt = os.path.join(save_dir, 'gt')
    os.makedirs(os.path.expanduser(save_dir), exist_ok=True)
    os.makedirs(os.path.expanduser(save_dir_gt), exist_ok=True)

    ddim_stochastic = partial(diffusion.ddim_sample_loop, eta=1.)
    sample_fn = (
        diffusion.p_sample_loop if not config.sampling.use_ddim else ddim_stochastic
    )

    logger.log(f"loading midi from test set cls {args.class_label} to edit...")
    val_data = load_data(
        data_dir=args.data_dir + "_test_cls_" + str(args.class_label) + ".csv",
        batch_size=args.batch_size,
        class_cond=True,
        image_size=gen_shape[2],
        rule=None,
    )

    logger.log("sampling...")
    count_samples = 0
    # target rules are dummy for get editing table
    target_rules = vars(config.target_rules)
    if 'vertical_nd' in target_rules.keys():
        # create a new dummy rule name and delete the old names
        target_rules['note_density_pixel'] = None
        target_rules.pop('vertical_nd')
        target_rules.pop('horizontal_nd')
    model_kwargs = {"rule": {k: v for k, v in target_rules.items()}}
    if args.class_cond:
        model_kwargs["y"] = classes

    all_results = pd.DataFrame()

    while count_samples < args.num_samples:
        with th.no_grad():
            gt, extra = next(val_data)
            gt = gt.to(dist_util.dev())
            for rule_name in target_rules.keys():
                target_rule = _extract_rule(rule_name, gt)
                model_kwargs["rule"][rule_name] = target_rule

        sample = sample_fn(
            model_fn_used,
            gen_shape,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            cond_fn=cond_fn_used,
            # None for NN(z_0), embed_model for rule(decoder(z_0))
            embed_model=embed_model if config.guidance.vae else None,
            scale_factor=args.scale_factor,
            guidance_kwargs=config.guidance,
            scg_kwargs=vars(config.scg) if config.guidance.scg else None,
            t_end=config.sampling.t_end,
            record=args.record,
            progress=True
        )
        sample = midi_util.decode_sample_for_midi(sample, embed_model=embed_model,
                                                  scale_factor=args.scale_factor, threshold=-0.95)
        arr = sample.cpu().numpy()
        arr = arr.transpose(0, 3, 1, 2)
        gt = ((gt + 1) * 63.5).clamp(0, 127).to(th.uint8)
        arr_gt = gt.cpu().numpy()
        if args.save_files:
            if args.class_cond:
                label_arr = classes.cpu().numpy()
                midi_util.save_piano_roll_midi(arr, save_dir, args.fs, y=label_arr, save_ind=count_samples)
                midi_util.save_piano_roll_midi(arr_gt, save_dir_gt, args.fs, y=label_arr, save_ind=count_samples)
            else:
                midi_util.save_piano_roll_midi(arr, save_dir, args.fs, save_ind=count_samples)
                midi_util.save_piano_roll_midi(arr_gt, save_dir_gt, args.fs, save_ind=count_samples)

        # test distance between generated samples and target
        generated_samples = th.from_numpy(arr) / 63.5 - 1
        results = midi_util.eval_rule_loss(generated_samples, model_kwargs["rule"])
        all_results = pd.concat([all_results, results], ignore_index=True)
        # save every step
        if args.save_files:
            all_results.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
        count_samples += args.batch_size

    if args.save_files:
        all_results.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
        # Create the DataFrame for loss_stats
        loss_columns = [col for col in all_results.columns if '.loss' in col]
        rows = []
        for col in loss_columns:
            rows.append({'Attr': col, 'Mean': all_results[col].mean(), 'Std': all_results[col].std()})
        loss_stats = pd.DataFrame(rows, columns=['Attr', 'Mean', 'Std'])
        loss_stats.to_csv(os.path.join(save_dir, 'summary.csv'))
        print(loss_stats)

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        project="music-sampling",
        dir="",
        data_dir="",  # use to load in val data to extract rule
        config_path="",
        model="",  # DiT model names
        model_path="",
        clip_denoised=False,
        num_samples=128,
        batch_size=16,
        scale_factor=1.,
        fs=12.5,
        num_classes=0,
        class_label=1,  # class to generate
        cfg=False,
        w=4.,  # for cfg
        classifier_scale=1.0,
        record=False,
        save_files=True,
        training=False,  # not training, so don't need to create more folders than needed
        port=None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
