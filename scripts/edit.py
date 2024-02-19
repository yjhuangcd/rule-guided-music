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
from guided_diffusion.midi_util import (
    VERTICAL_ND_BOUNDS, VERTICAL_ND_CENTER, HORIZONTAL_ND_BOUNDS, HORIZONTAL_ND_CENTER,
    get_full_piano_roll
)
from guided_diffusion.dit import DiT_models
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_diffusion,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.gaussian_diffusion import _encode, _extract_rule
from guided_diffusion.pr_datasets_all import load_data
from load_utils import load_model
import diff_collage as dc
from guided_diffusion.condition_functions import (
    model_fn, dc_model_fn, composite_nn_zt, composite_rule)
from functools import partial
import matplotlib.pyplot as plt
import pretty_midi

plt.rcParams["figure.figsize"] = (20, 3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def main():
    args = create_argparser().parse_args()
    root_dir = 'edit_demo/'
    args.dir = root_dir + os.path.splitext(args.config_path.split(root_dir)[-1])[0] + f'_cls_{args.class_label}'

    comm = dist_util.setup_dist(port=args.port)
    logger.configure(args=args, comm=comm)
    config = midi_util.load_config(args.config_path)
    if config.sampling.use_ddim:
        args.timestep_respacing = config.sampling.timestep_respacing

    logger.log("creating model and diffusion...")
    model = DiT_models[args.model](
        input_size=args.image_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        learn_sigma=args.learn_sigma,
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

    # create embed model
    if args.vae is not None:
        embed_model = load_model(args.vae, args.vae_path)
        embed_model.to(dist_util.dev())
        embed_model.eval()
    else:
        embed_model = None

    cond_fn_config = config.guidance.cond_fn
    if config.guidance.nn:
        logger.log("loading classifier...")
        classifier_config = cond_fn_config.classifiers
        num_classifiers = len(classifier_config.names)
        classifiers = []
        for i in range(num_classifiers):
            classifier = DiT_models[classifier_config.names[i]](
                # classifier trained on latents, so has the same img size as diffusion
                input_size=args.image_size,
                in_channels=args.in_channels,
                num_classes=classifier_config.num_classes[i],
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
        def eps_fn(x, t, y=None):
            # since our backbone takes 128x16 as input
            return model(x.permute(0, 1, 3, 2), t, y=y).permute(0, 1, 3, 2)

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

    edit_kwargs = vars(config.edit)
    edit_kwargs["l_start_pix"] = edit_kwargs["l_start"] * 8
    edit_kwargs["l_end_pix"] = edit_kwargs["l_end"] * 8
    source = getattr(config.edit, 'source', None)
    if source == 'dataset':
        logger.log(f"loading midi from test set cls {args.class_label} to edit...")
        val_data = load_data(
            data_dir=args.data_dir + "_test_cls_" + str(args.class_label) + ".csv",
            batch_size=args.batch_size,
            class_cond=True,
            image_size=gen_shape[2] * 8,
            rule=None,
        )
        gt, extra = next(val_data)
        gt = gt.to(dist_util.dev())
    else:
        midi_data = pretty_midi.PrettyMIDI(source)
        gt = get_full_piano_roll(midi_data, fs=args.fs)
        gt = th.from_numpy(gt).float()[None] / 63.5 - 1
        gt = F.pad(gt, (0, gen_shape[2] * 8 - gt.shape[3]), "constant", -1)
        gt = gt.to(dist_util.dev())

    gt_latent = _encode(gt, embed_model, scale_factor=args.scale_factor)
    mask = th.ones_like(gt_latent)
    mask[:, :, edit_kwargs["l_start"]:edit_kwargs["l_end"], :] = 0.
    edit_kwargs["gt"] = gt_latent
    edit_kwargs["mask"] = mask

    logger.log("sampling...")

    with th.no_grad():
        model_kwargs = {"rule": {}}
        target_rules = vars(config.target_rules)
        gt_partial = gt[:, :, :, edit_kwargs["l_start"]*8:edit_kwargs["l_end"]*8]
        for rule_name, val in target_rules.items():
            if 'horizontal' in rule_name:
                continue
            # generate a different target for nd, generate the same target for chord
            elif 'vertical' in rule_name:
                hr_nd = target_rules[rule_name.replace('vertical', 'horizontal')]
                if '_hr_' in rule_name:
                    str_hr_scale = rule_name.split('_hr_')[-1]
                    horizontal_scale = int(str_hr_scale)
                    rule_name = f'note_density_hr_{str_hr_scale}'
                else:
                    horizontal_scale = 5
                    rule_name = 'note_density'
                # need orig_rule for all cases because want to record orig_rule
                orig_rule = _extract_rule(rule_name, gt_partial)
                if len(orig_rule.shape) == 1:
                    # unsqueeze the first dimension of batch_size = 1
                    orig_rule = orig_rule.reshape(1, -1)

                # if not given target or target is to shift extracted nd
                if isinstance(val, int) or val is None:
                    # need to compute class to shift
                    vt_bounds = th.tensor(VERTICAL_ND_BOUNDS).to(dist_util.dev())
                    hr_bounds = th.tensor(HORIZONTAL_ND_BOUNDS).to(dist_util.dev()) / horizontal_scale
                    vt_center = th.tensor(VERTICAL_ND_CENTER).to(dist_util.dev())
                    hr_center = th.tensor(HORIZONTAL_ND_CENTER).to(dist_util.dev()) / horizontal_scale
                    if isinstance(val, int):
                        vertical_rand = val
                        horizontal_rand = hr_nd
                    else:
                        # randomly shift nd
                        vertical_range = 1
                        horizontal_range = 1
                        vertical_rand = th.randint(-vertical_range, vertical_range + 1,
                                                   size=(orig_rule.shape[0], 1), device=orig_rule.device)
                        horizontal_rand = th.randint(-horizontal_range, horizontal_range + 1,
                                                     size=(orig_rule.shape[0], 1), device=orig_rule.device)
                    total_length = orig_rule.shape[-1]
                    vt_nd_classes = (th.bucketize(orig_rule[:, :total_length // 2], vt_bounds) + vertical_rand)
                    hr_nd_classes = (th.bucketize(orig_rule[:, total_length // 2:], hr_bounds) + horizontal_rand)
                    vt_nd_val = vt_center[vt_nd_classes.clamp_(min=0, max=7)]
                    hr_nd_val = hr_center[hr_nd_classes.clamp_(min=0, max=7)]
                    target_rule = th.concat((vt_nd_val, hr_nd_val), dim=-1)
                else:
                    # use given nd
                    hr_nd_rescale = [x / horizontal_scale for x in hr_nd]
                    nd_val = val + hr_nd_rescale
                    target_rule = th.tensor(nd_val, device=dist_util.dev())
            elif 'pitch' in rule_name and val is not None:
                orig_rule = _extract_rule(rule_name, gt_partial)
                val = th.tensor(val, device=dist_util.dev())
                target_rule = val / (th.sum(val) + 1e-12)
            else:
                orig_rule = _extract_rule(rule_name, gt_partial)
                if val is not None:
                    target_rule = th.tensor(val, device=dist_util.dev())
                else:
                    target_rule = _extract_rule(rule_name, gt_partial)
            if source == 'dataset':
                if len(target_rule.shape) == 1:
                    target_rule = target_rule.reshape(1, -1).repeat(args.batch_size, 1)
                model_kwargs["rule"][rule_name] = target_rule
            else:
                # if given only one source, generate multiple variations
                model_kwargs["rule"][rule_name] = target_rule.repeat(args.batch_size, 1)

    if args.class_cond:
        model_kwargs["y"] = classes

    all_results = pd.DataFrame()

    count_samples = 0
    while count_samples < args.num_samples:
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
            edit_kwargs=edit_kwargs,
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
        # only take editable part to compute rule loss
        generated_samples = generated_samples[:, :, :, edit_kwargs["l_start_pix"]:edit_kwargs["l_end_pix"]]
        results = midi_util.eval_rule_loss(generated_samples, model_kwargs["rule"])
        # save original rules
        orig_rule_dict = {}
        for rule_name in model_kwargs["rule"].keys():
            orig_rule_dict[rule_name + '.orig_rule'] = orig_rule.cpu().tolist()
        orig_rule_df = pd.DataFrame(orig_rule_dict)
        results = pd.concat([orig_rule_df, results], axis=1)
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
        data_dir="",   # use to load in val data to extract rule
        config_path="",
        model="DiTRotary_XL_8",  # DiT model names
        model_path="",
        vae="kl/f8-all-onset",
        vae_path="taming-transformers/checkpoints/all_onset/epoch_14.ckpt",
        clip_denoised=False,
        num_samples=128,
        batch_size=16,
        scale_factor=1.,
        fs=100,
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
