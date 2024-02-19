import copy
import functools
import os
import os.path as osp
import numpy as np
import math

import blobfile as bf
import torch as th
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, midi_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from taming.modules.distributions.distributions import DiagonalGaussianDistribution

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        eval_model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        embed_model=None,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        eval_data=None,
        eval_interval=-1,
        eval_sample_batch_size=16,
        total_num_gpus=1,   # training is run on how many gpus, used to distribute classes on each gpu
        eval_sample_use_ddim=True,
        eval_sample_clip_denoised=True,
        in_channels=1,
        fs=100,
        pedal=False,        # whether decode with pedal as the second channel
        scale_factor=1.,
        num_classes=0,   # whether to use class_cond in sampling
        microbatch_encode=-1,
        encode_rep=4,
        shift_size=4,   # shift_size when generating time shifted sampels from an encoding
    ):
        self.model = model
        self.eval_model = eval_model
        self.embed_model = embed_model
        self.scale_factor = scale_factor
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.microbatch_encode = microbatch_encode
        self.encode_rep = encode_rep
        self.batch_size = self.batch_size // self.encode_rep    # effective batch size
        self.microbatch = self.microbatch // self.encode_rep
        self.shift_size = shift_size  # need to be compatible with encode_rep
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        # eval
        self.eval_data = eval_data
        self.eval_interval = eval_interval
        self.total_num_gpus = total_num_gpus
        self.eval_sample_batch_size = eval_sample_batch_size // self.total_num_gpus
        self.eval_sample_use_ddim = eval_sample_use_ddim
        self.eval_sample_clip_denoised = eval_sample_clip_denoised
        self.in_channels = in_channels
        self.fs = fs
        self.pedal = pedal
        self.num_classes = num_classes

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                 dist_util.load_state_dict(
                     resume_checkpoint, map_location=dist_util.dev()
                 )
             )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                 ema_checkpoint, map_location=dist_util.dev()
             )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            dist.barrier()
            self.run_step(batch, cond)
            if self.eval_data is not None and self.step % self.eval_interval == 0:
                batch_eval, cond_eval = next(self.eval_data)
                self.run_step_eval(batch_eval, cond_eval)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0 and self.step != 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def run_step_eval(self, batch, cond):
        with th.no_grad():
            # load in ema_params for eval_model in cpu, then move to gpu
            # only use the first ema rate if there are multiple ema rate
            ema_state_dict = self.mp_trainer.master_params_to_state_dict(self.ema_params[0])
            # ema_state_dict_cpu = {k: v.cpu() for k, v in ema_state_dict.items()}
            # self.eval_model.load_state_dict(ema_state_dict_cpu)
            self.eval_model.load_state_dict(ema_state_dict)
            # self.eval_model.to(dist_util.dev())
            if self.use_fp16:
                self.eval_model.convert_to_fp16()
            self.eval_model.eval()
            for i in range(0, batch.shape[0], self.microbatch):
                micro = batch[i: i + self.microbatch].to(dist_util.dev())
                if self.embed_model is not None:
                    micro = get_kl_input(micro, microbatch=self.microbatch_encode,
                                         model=self.embed_model, scale_factor=self.scale_factor,
                                         shift_size=self.shift_size)
                micro_cond = {
                    k: v[i: i + self.microbatch].repeat_interleave(self.encode_rep).to(dist_util.dev())
                    for k, v in cond.items()
                }
                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.eval_model,
                    micro,
                    t,
                    model_kwargs=micro_cond,
                )
                losses = compute_losses()
                log_loss_dict(
                    self.diffusion, t, {'eval_'+k: v * weights for k, v in losses.items()}
                )
            if self.eval_sample_batch_size > 0 and self.step != 0:
            # if True:
                model_kwargs = {}
                if self.num_classes > 0:
                    # classes = th.randint(
                    #     low=0, high=self.num_classes, size=(self.eval_sample_batch_size,), device=dist_util.dev()
                    # )
                    # balance generated classes
                    rank = dist.get_rank()
                    samples_per_class = math.ceil(self.eval_sample_batch_size * self.total_num_gpus / self.num_classes)
                    label_start = rank * self.eval_sample_batch_size // samples_per_class
                    label_end = math.ceil((rank + 1) * self.eval_sample_batch_size / samples_per_class)
                    classes = th.arange(label_start, label_end, dtype=th.int, device=dist_util.dev()).repeat_interleave(samples_per_class)
                    model_kwargs["y"] = classes[:self.eval_sample_batch_size]
                all_images = []
                all_labels = []
                image_size_h = micro.shape[-2]
                image_size_w = micro.shape[-1]
                sample_fn = (
                    self.diffusion.p_sample_loop if not self.eval_sample_use_ddim else self.diffusion.ddim_sample_loop
                )
                sample = sample_fn(
                    self.eval_model,
                    (self.eval_sample_batch_size, self.in_channels, image_size_h, image_size_w),
                    # (4, self.in_channels, image_size_h, image_size_w),
                    clip_denoised=self.eval_sample_clip_denoised,
                    model_kwargs=model_kwargs,
                    progress=True
                )
                ##### debug
                # sample = micro
                sample = midi_util.decode_sample_for_midi(sample, embed_model=self.embed_model,
                                                          scale_factor=self.scale_factor, threshold=-0.95)

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
                if self.num_classes > 0:
                    gathered_labels = [
                        th.zeros_like(model_kwargs["y"]) for _ in range(dist.get_world_size())
                    ]
                    dist.all_gather(gathered_labels, model_kwargs["y"])
                    all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

                arr = np.concatenate(all_images, axis=0)
                if arr.shape[-1] == 1:  # no pedal, need shape B x 128 x 1024
                    arr = arr.squeeze(axis=-1)
                else:   # with pedal, need shape: B x 2 x 128 x 1024
                    arr = arr.transpose(0, 3, 1, 2)
                if self.num_classes > 0:
                    label_arr = np.concatenate(all_labels, axis=0)
                save_dir = osp.join(get_blob_logdir(), "samples", "iter_" + str(self.step + self.resume_step))
                os.makedirs(os.path.expanduser(save_dir), exist_ok=True)
                if dist.get_rank() == 0:
                    if self.num_classes > 0:
                        midi_util.save_piano_roll_midi(arr, save_dir, self.fs, y=label_arr)
                    else:
                        midi_util.save_piano_roll_midi(arr, save_dir, self.fs)
                dist.barrier()
            # # put the model on cpu to prepare for next loading
            # self.eval_model.to("cpu")

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            if self.embed_model is not None:
                micro = get_kl_input(micro, microbatch=self.microbatch_encode,
                                     model=self.embed_model, scale_factor=self.scale_factor,
                                     shift_size=self.shift_size)
            micro_cond = {
                k: v[i : i + self.microbatch].repeat_interleave(self.encode_rep).to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= self.batch_size
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)
            # # keep gpu mem constant?
            # del losses

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), "checkpoints", filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), "checkpoints", f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


@th.no_grad()
def get_kl_input(batch, microbatch=-1, model=None, scale_factor=1., recombine=True, shift_size=4):
    # here microbatch should be outer microbatch // encode_rep
    if microbatch < 0:
        microbatch = batch.shape[0]
    full_z = []
    image_size_h = batch.shape[-2]
    image_size_w = batch.shape[-1]
    seq_len = image_size_w // image_size_h
    for i in range(0, batch.shape[0], microbatch):
        micro = batch[i : i + microbatch].to(dist_util.dev())
        # encode each 1s and concatenate
        micro = th.chunk(micro, seq_len, dim=-1)  # B x C x H x W
        micro = th.concat(micro, dim=0)  # 1st second for all batch, 2nd second for all batch, ...
        micro = model.encode_save(micro, range_fix=False)
        posterior = DiagonalGaussianDistribution(micro)
        # z = posterior.sample()
        z = posterior.mode()
        z = th.concat(th.chunk(z, seq_len, dim=0), dim=-1)
        z = z.permute(0, 1, 3, 2)
        full_z.append(z)
    full_z = th.concat(full_z, dim=0)    # B x 4 x (15x16), 16
    if recombine:   # if not using microbatch, then need to use recombination of tokens
        # unfold: dimension, size, step
        full_z = full_z.unfold(2, 8*16, 16*shift_size).permute(0, 2, 1, 4, 3)   # (B x encode_rep) x 4 x 128 x 16
        full_z = full_z.contiguous().view(-1, 4, 8*16, 16)     # B x 4 x 128 x 16
    return (full_z * scale_factor).detach()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
