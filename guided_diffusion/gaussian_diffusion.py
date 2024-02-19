"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import os

import math

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from .midi_util import save_piano_roll_midi
from music_rule_guidance.rule_maps import FUNC_DICT, LOSS_DICT
from collections import defaultdict
import torch.nn.functional as F
import multiprocessing
from functools import partial

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20, 3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == 'stable-diffusion':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * math.sqrt(0.00085)
        beta_end = scale * math.sqrt(0.012)
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        ) ** 2
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None,
        cond_fn=None, embed_model=None, edit_kwargs=None,
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param cond_fn: log p(y|x), to maximize
        :param embed_model: contains encoder and decoder
        :param edit_kwargs: replacement-based conditioning
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if edit_kwargs is not None:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
            replaced_x0 = edit_kwargs["mask"] * edit_kwargs["gt"] + (1 - edit_kwargs["mask"]) * pred_xstart
            model_output = self._predict_eps_from_xstart(x_t=x, t=t, pred_xstart=replaced_x0)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None, guidance_kwargs=None,
                       model=None, embed_model=None, edit_kwargs=None, scale_factor=1.,
                       record=False):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        If dps=True, use diffusion posterior sampling, cond_fn is log p(y|x_0)
        instead of the grad of it. Need to use model (eps) and embed_model.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        dps = True if guidance_kwargs.method == 'dps' else False
        if not dps:
            if edit_kwargs is None:
                gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
                new_mean = (
                    p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
                )
            else:
                # only compute gradient on editable latents, since rule is only on editable length
                x = x[:, :, edit_kwargs["l_start"]:edit_kwargs["l_end"], :]
                gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
                new_mean = p_mean_var["mean"].float()
                new_mean[:, :, edit_kwargs["l_start"]:edit_kwargs["l_end"], :] += (
                    p_mean_var["variance"] * gradient.float())
        else:
            assert model is not None
            step_size = guidance_kwargs.step_size
            with th.enable_grad():
                xt = x.detach().requires_grad_(True)
                eps = model(xt, self._scale_timesteps(t), **model_kwargs)
                pred_xstart = self._predict_xstart_from_eps(xt, t, eps)
                # If vae is not None, and not dps_nn, i.e. using dps rule
                if embed_model is not None and not guidance_kwargs.nn:
                    pred_xstart = _decode(pred_xstart, embed_model, scale_factor=scale_factor)
                if record:
                    pred_xstart.retain_grad()
                if edit_kwargs is not None:
                    # only check condition on the editable part
                    pred_xstart = pred_xstart[:, :, edit_kwargs["l_start"]:edit_kwargs["l_end"], :]
                log_probs = cond_fn(pred_xstart, self._scale_timesteps(t), **model_kwargs)
                gradient = th.autograd.grad(log_probs.sum(), xt)[0]

            # check if x_0 space works
            if record:
                pred_xstart_up = pred_xstart + pred_xstart.grad
                log_probs_up = cond_fn(pred_xstart_up, self._scale_timesteps(t), **model_kwargs)
                # record gradient difference
                cur_grad_diff = (self.prev_gradient_single - gradient).reshape(x.shape[0], -1).norm(dim=-1)
                prev_gradient_norm = self.prev_gradient_single.reshape(x.shape[0], -1).norm(dim=-1)
                if prev_gradient_norm.mean() > 1e-5:
                    self.grad_norm.append(prev_gradient_norm.mean().item())
                    cur_grad_diff = cur_grad_diff / prev_gradient_norm
                    self.gradient_diff.append(cur_grad_diff.mean().item())
                self.prev_gradient_single = gradient
                self.log_probs.append((log_probs.mean().item()))

            gradient = gradient / th.sqrt(-log_probs.view(x.shape[0], 1, 1, 1) + 1e-12)
            # gradient = gradient / (-log_probs.view(x.shape[0], 1, 1, 1) + 1e-12)

            if edit_kwargs is None:
                new_mean = (
                    p_mean_var["mean"].float() + step_size * gradient.float()
                )
            else:
                new_mean = p_mean_var["mean"].float()
                new_mean[:, :, edit_kwargs["l_start"]:edit_kwargs["l_end"], :] += step_size * gradient.float()

            # check whether moved towards good direction om z space
            if record:
                eps = model(xt + step_size * gradient.float(), self._scale_timesteps(t), **model_kwargs)
                pred_xstart_2 = self._predict_xstart_from_eps(xt, t, eps)
                pred_xstart_2 = _decode(pred_xstart_2, embed_model, scale_factor=scale_factor)
                log_probs_2 = cond_fn(pred_xstart_2, self._scale_timesteps(t), **model_kwargs)

        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def scg_sample(self,
                    model,
                    t,
                    mean_pred,
                    g_coeff,
                    embed_model,
                    scale_factor,
                    model_kwargs=None,
                    scg_kwargs=None,
                    edit_kwargs=None,
                    dc_kwargs=None,
                    record=False,
                    record_freq=100):
        """
        Sample N x_{t-1} from x_t and select the best one.
        """
        # mean_pred = p_mean_var["mean"]
        # g_coeff = th.exp(0.5 * p_mean_var["log_variance"])
        num_samples = scg_kwargs["num_samples"]
        sample = mean_pred.unsqueeze(dim=0)
        sample = sample.expand(num_samples, *mean_pred.shape).contiguous()
        noise = th.randn_like(sample)
        sample = sample + g_coeff * noise
        sample = sample.view(-1, *mean_pred.shape[1:])
        t = t.repeat(num_samples)
        # it's fine to use different target for different samples, expand and repeat match with each other (012012)
        cloned_model_kwargs = {"y": model_kwargs["y"].repeat(num_samples)}
        eps = model(sample, self._scale_timesteps(t), **cloned_model_kwargs)
        pred_xstart = self._predict_xstart_from_eps(sample, t, eps)
        if edit_kwargs is not None:
            # only decode editable part
            pred_xstart = pred_xstart[:, :, edit_kwargs["l_start"]:edit_kwargs["l_end"], :]
        if embed_model is not None:
            pred_xstart = _decode(pred_xstart, embed_model, scale_factor=scale_factor)

        if dc_kwargs is None or dc_kwargs.base <= 0:
            if record:
                # create dictionary to record the loss for each rule
                each_loss = {}
            # work with multiple rules, model_kwargs["rule"] is a dict that contains rule_name: target
            total_log_prob = 0
            for rule_name, rule_target in model_kwargs["rule"].items():
                gen_rule = _extract_rule(rule_name, pred_xstart)
                y_ = rule_target.repeat(num_samples, 1)
                log_prob = - LOSS_DICT[rule_name](gen_rule, y_)
                if record:
                    each_loss[rule_name] = -log_prob.view(num_samples, -1)
                total_log_prob += log_prob * scg_kwargs.get(rule_name, 1.)
            total_log_prob = total_log_prob.view(num_samples, -1)
            max_ind = total_log_prob.argmax(dim=0)

            # softmax (need to reweight to get unit var otherwise goes to empty rolls)
            # weight = F.softmax(total_log_prob * 1., dim=0)
            # var = (weight ** 2).sum(dim=0)
            # avg_noise = (noise * weight[..., None, None, None]).sum(dim=0) / th.sqrt(var)[..., None, None, None]
            # # not adding dw
            # sample = mean_pred + g_coeff * avg_noise
            # # add dw
            # dw = th.randn_like(p_mean_var["mean"])
            # sample = mean_pred + g_coeff * (avg_noise + dw)

            # take argmax
            sample = sample.view(num_samples, *mean_pred.shape)
            sample = sample[max_ind, th.arange(mean_pred.shape[0])]

            # take argmax, and add dw
            # noise = noise.view(num_samples, *p_mean_var["mean"].shape)
            # best_noise = noise[max_ind, th.arange(p_mean_var["mean"].shape[0])]
            # dw = th.randn_like(p_mean_var["mean"])
            # sample = p_mean_var["mean"] + th.exp(0.5 * p_mean_var["log_variance"]) * (best_noise + dw)

        else:
            # Assuming base length in x0 is only controlled by the corresponding location in xt
            # (doesn't hold, but maybe can approximate because of cond ind)
            sample = sample.view(num_samples, *mean_pred.shape)
            sub_samples = []
            total_length = pred_xstart.shape[-1]
            start_inds = th.arange(0, total_length, dc_kwargs.base*8)
            rule_base = dc_kwargs.base // 16   # number of rules under the base length
            for i, start_ind in enumerate(start_inds):
                end_ind = min(start_ind+dc_kwargs.base*8, total_length)
                pred_xstart_cur = pred_xstart[:, :, :, start_ind: end_ind]
                total_log_prob = 0
                for rule_name, rule_target in model_kwargs["rule"].items():
                    gen_rule = _extract_rule(rule_name, pred_xstart_cur)
                    if rule_name == 'note_density':
                        half = rule_target.shape[-1] // 2
                        vt_nd_target = rule_target[:, :half][:, i*rule_base: min((i+1)*rule_base, half)]
                        hr_nd_target = rule_target[:, half:][:, i*rule_base: min((i+1)*rule_base, half)]
                        rule_target = th.concat((vt_nd_target, hr_nd_target), dim=-1)
                    elif 'chord' in rule_name:
                        rule_length = rule_target.shape[-1]
                        rule_target = rule_target[:, i*rule_base: min((i+1)*rule_base, rule_length)]
                    y_ = rule_target.repeat(num_samples, 1)
                    log_prob = - LOSS_DICT[rule_name](gen_rule, y_)
                    total_log_prob += log_prob * scg_kwargs.get(rule_name, 1.)
                total_log_prob = total_log_prob.view(num_samples, -1)
                max_ind = total_log_prob.argmax(dim=0)
                # take argmax on num_sample x batch_size x 4 x 256 x 16
                sub_sample = sample[max_ind, th.arange(mean_pred.shape[0]), :, start_ind//8: end_ind//8]
                sub_samples.append(sub_sample)
            sample = th.concat(sub_samples, dim=-2)

        if record:
            for rule_name, loss in each_loss.items():
                current_loss = loss[max_ind, th.arange(mean_pred.shape[0])][0].item()
                self.each_loss[rule_name].append((t[0].item(), current_loss))
            max_log_prob = total_log_prob[max_ind, th.arange(mean_pred.shape[0])][0].item()
            # record log_prob
            self.log_probs.append((t[0].item(), max_log_prob))
            # record loss std
            self.loss_std.append((t[0].item(), total_log_prob.std().item()))
            # record loss range
            self.loss_range.append((t[0].item(), (max_log_prob - total_log_prob.min()).abs().item()))
            # record gradient difference
            noise = noise.view(num_samples, *mean_pred.shape)
            gradient = noise[max_ind, th.arange(mean_pred.shape[0])]
            cur_grad_diff = (self.prev_gradient_single - gradient).reshape(sample.shape[0], -1).norm(dim=-1)
            prev_gradient_norm = self.prev_gradient_single.reshape(sample.shape[0], -1).norm(dim=-1)
            if prev_gradient_norm.mean() > 1e-5:
                self.grad_norm.append(prev_gradient_norm.mean().item())
                cur_grad_diff = cur_grad_diff / prev_gradient_norm
                self.gradient_diff.append(cur_grad_diff.mean().item())
            self.prev_gradient_single = gradient
            if (t[0] + 1) % record_freq == 0:
                pred_xstart = pred_xstart.view(num_samples, -1, *pred_xstart.shape[1:])
                pred_xstart = pred_xstart[max_ind, th.arange(mean_pred.shape[0])]
                pred_xstart[pred_xstart <= -0.95] = -1.  # heuristic thresholding the background
                pred_xstart = ((pred_xstart + 1) * 63.5).clamp(0, 127).to(th.uint8)
                self.inter_piano_rolls.append(pred_xstart.cpu())

                # plot loss distribution
                if len(model_kwargs["rule"].keys()) <= 1:
                    plt.figure(figsize=(4, 3))
                    total_log_prob = total_log_prob.view(-1).cpu()
                    plt.bar(range(len(total_log_prob)), -total_log_prob)
                    plt.xlabel('choice')
                    plt.ylabel('loss')
                    plt.title(f't={t[0]+1}')
                    plt.tight_layout()
                    plt.savefig(f'loggings/debug/t={t[0]+1}.png')
                    plt.show()
        return sample

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        embed_model=None,
        scale_factor=1.,
        guidance_kwargs=None,
        scg_kwargs=None,
        edit_kwargs=None,
        record=False,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        if guidance_kwargs is not None:
            if guidance_kwargs.schedule:
                t_start = guidance_kwargs.t_start
                t_end = guidance_kwargs.t_end
                interval = guidance_kwargs.interval
                use_guidance = guide_schedule(t, t_start, t_end, interval)
            else:
                use_guidance = True
        else:
            use_guidance = False
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            embed_model=embed_model,
            edit_kwargs=edit_kwargs,
        )

        # if use scg guidance, then schedule only applies to scg sampling
        if cond_fn is not None and (use_guidance or scg_kwargs is not None):
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs,
                guidance_kwargs=guidance_kwargs, model=model, embed_model=embed_model,
                edit_kwargs=edit_kwargs, scale_factor=scale_factor
            )

        if scg_kwargs is None:
            noise = th.randn_like(x)
            nonzero_mask = (
                (t > self.t_end).float().view(-1, *([1] * (len(x.shape) - 1)))
            )  # no noise when t == t_end (0 if not early stopping)
            sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        else:   # scg search (greedy)
            if t[0] > self.t_end:
                mean_pred = out["mean"]
                g_coeff = th.exp(0.5 * out["log_variance"])
                if use_guidance:
                    dc_kwargs = getattr(guidance_kwargs, 'dc', None)
                    sample = self.scg_sample(model, t, mean_pred, g_coeff, embed_model, scale_factor,
                                              model_kwargs=model_kwargs, scg_kwargs=scg_kwargs,
                                              edit_kwargs=edit_kwargs, dc_kwargs=dc_kwargs, record=record)
                else:
                    sample = mean_pred + g_coeff * th.randn_like(x)
                    if record:
                        eps = model(sample, self._scale_timesteps(t), **model_kwargs)
                        pred_xstart = self._predict_xstart_from_eps(sample, t, eps)
                        pred_xstart = _decode(pred_xstart, embed_model, scale_factor=scale_factor)
                        if len(model_kwargs["rule"].keys()) <= 1:
                            # only record for individual rule to save time
                            total_log_prob = 0
                            for rule_name, rule_target in model_kwargs["rule"].items():
                                gen_rule = _extract_rule(rule_name, pred_xstart)
                                log_prob = - LOSS_DICT[rule_name](gen_rule, rule_target)
                                total_log_prob += log_prob.mean().item() * scg_kwargs.get(rule_name, 1.)
                            self.log_probs.append((t[0].item(), total_log_prob))
                        if (t[0] + 1) % 100 == 0:
                            pred_xstart[pred_xstart <= -0.95] = -1.  # heuristic thresholding the background
                            pred_xstart = ((pred_xstart + 1) * 63.5).clamp(0, 127).to(th.uint8)
                            self.inter_piano_rolls.append(pred_xstart.cpu())
            else:
                sample = out["mean"]

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        t_end=0,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        embed_model=None,
        scale_factor=1.,
        guidance_kwargs=None,
        scg_kwargs=None,
        edit_kwargs=None,
        record=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param t_end: early stopping for the sampling process
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        self.t_end = t_end
        if record:
            self.prev_gradient_single = th.zeros(shape, device=device)
            self.gradient_diff = []
            self.grad_norm = []
            self.log_probs = []
            # record loss for each rule
            self.each_loss = defaultdict(list)
            self.inter_piano_rolls = []
            self.loss_std = []
            self.loss_range = []
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            t_end=t_end,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            embed_model=embed_model,
            scale_factor=scale_factor,
            guidance_kwargs=guidance_kwargs,
            scg_kwargs=scg_kwargs,
            edit_kwargs=edit_kwargs,
            record=record,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        t_end=0,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        embed_model=None,
        scale_factor=1.,
        guidance_kwargs=None,
        scg_kwargs=None,
        edit_kwargs=None,
        record=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        elif edit_kwargs is not None:
            t = th.tensor([edit_kwargs["noise_level"]-1] * shape[0], device=device)
            alpha_cumprod = _extract_into_tensor(self.alphas_cumprod, t, shape)
            img = th.sqrt(alpha_cumprod) * edit_kwargs["gt"] + th.sqrt((1 - alpha_cumprod)) * th.randn(*shape, device=device)
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        if t_end:
            indices = indices[:-t_end]
        if edit_kwargs is not None:
            t_start = self.num_timesteps - edit_kwargs["noise_level"]
            indices = indices[t_start:]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    embed_model=embed_model,
                    scale_factor=scale_factor,
                    guidance_kwargs=guidance_kwargs,
                    scg_kwargs=scg_kwargs,
                    edit_kwargs=edit_kwargs,
                    record=record,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
        embed_model=None,
        scale_factor=1.,
        guidance_kwargs=None,
        edit_kwargs=None,
        scg_kwargs=None,
        record=False,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        if guidance_kwargs is not None:
            if guidance_kwargs.schedule:
                t_start = guidance_kwargs.t_start
                t_end = guidance_kwargs.t_end
                interval = guidance_kwargs.interval
                use_guidance = guide_schedule(t, t_start, t_end, interval)
            else:
                use_guidance = True
        else:
            use_guidance = False
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            embed_model=embed_model,
            edit_kwargs=edit_kwargs,
        )
        if cond_fn is not None and use_guidance:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        if scg_kwargs is None:
            noise = th.randn_like(x)
            nonzero_mask = (
                (t != self.t_end).float().view(-1, *([1] * (len(x.shape) - 1)))
            )  # no noise when t == t_end (0 if not early stopping)
            sample = mean_pred + nonzero_mask * sigma * noise
        else:
            if t[0] > self.t_end:
                g_coeff = sigma
                if use_guidance:  # tune according to ddim steps
                    dc_kwargs = getattr(guidance_kwargs, 'dc', None)
                    sample = self.scg_sample(self._wrap_model(model), t, mean_pred, g_coeff, embed_model, scale_factor,
                                              model_kwargs=model_kwargs, scg_kwargs=scg_kwargs, edit_kwargs=edit_kwargs,
                                              dc_kwargs=dc_kwargs, record=record, record_freq=10)
                else:
                    sample = mean_pred + g_coeff * th.randn_like(x)
                    if record:
                        eps = self._wrap_model(model)(sample, self._scale_timesteps(t), **model_kwargs)
                        pred_xstart = self._predict_xstart_from_eps(sample, t, eps)
                        pred_xstart = _decode(pred_xstart, embed_model, scale_factor=scale_factor)
                        total_log_prob = 0
                        for rule_name, rule_target in model_kwargs["rule"].items():
                            gen_rule = _extract_rule(rule_name, pred_xstart)
                            log_prob = - LOSS_DICT[rule_name](gen_rule, rule_target)
                            total_log_prob += log_prob.mean().item() * scg_kwargs.get(rule_name, 1.)
                        self.log_probs.append((t[0].item(), total_log_prob))

                        if (t[0] + 1) % 10 == 0:
                            pred_xstart[pred_xstart <= -0.95] = -1.  # heuristic thresholding the background
                            pred_xstart = ((pred_xstart + 1) * 63.5).clamp(0, 127).to(th.uint8)
                            self.inter_piano_rolls.append(pred_xstart.cpu())
            else:
                sample = mean_pred
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        t_end=0,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        embed_model=None,
        scale_factor=1.,
        guidance_kwargs=None,
        scg_kwargs=None,
        edit_kwargs=None,
        record=False,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        self.t_end = t_end
        if record:
            self.prev_gradient_single = th.zeros(shape, device=device)
            self.gradient_diff = []
            self.grad_norm = []
            self.log_probs = []
            self.inter_piano_rolls = []
            self.loss_std = []
            self.loss_range = []
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            t_end=t_end,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            embed_model=embed_model,
            scale_factor=scale_factor,
            guidance_kwargs=guidance_kwargs,
            scg_kwargs=scg_kwargs,
            edit_kwargs=edit_kwargs,
            record=record,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        t_end=0,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        embed_model=None,
        scale_factor=1.,
        guidance_kwargs=None,
        scg_kwargs=None,
        edit_kwargs=None,
        record=False,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        elif edit_kwargs is not None:
            t = th.tensor([edit_kwargs["noise_level"]-1] * shape[0], device=device)
            alpha_cumprod = _extract_into_tensor(self.alphas_cumprod, t, shape)
            img = th.sqrt(alpha_cumprod) * edit_kwargs["gt"] + th.sqrt((1 - alpha_cumprod)) * th.randn(*shape, device=device)
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        if t_end:
            indices = indices[:-t_end]
        if edit_kwargs is not None:
            t_start = self.num_timesteps - edit_kwargs["noise_level"]
            indices = indices[t_start:]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    embed_model=embed_model,
                    scale_factor=scale_factor,
                    guidance_kwargs=guidance_kwargs,
                    scg_kwargs=scg_kwargs,
                    edit_kwargs=edit_kwargs,
                    record=record,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def _decode(pred_zstart, embed_model, scale_factor=1., threshold=False):
    image_size_h = pred_zstart.shape[-2]
    image_size_w = pred_zstart.shape[-1]
    pred_zstart = pred_zstart / scale_factor
    sample = pred_zstart.permute(0, 1, 3, 2)
    sample = th.chunk(sample, image_size_h // image_size_w, dim=-1)  # B x C x H x W
    sample = th.concat(sample, dim=0)  # 1st second for all batch, 2nd second for all batch, ...
    sample = embed_model.decode(sample)
    pred_xstart = th.concat(th.chunk(sample, image_size_h // image_size_w, dim=0), dim=-1)
    if threshold:
        pred_xstart[pred_xstart <= -0.95] = -1.  # heuristic thresholding the background
    return pred_xstart


def _extract_rule(rule_name, pred_xstart):
    device = pred_xstart.device
    if 'chord' in rule_name:
        # Split tensor batch into smaller batches
        num_processes = 4
        pred_xstart = pred_xstart.cpu()
        pred_xstart_split = th.chunk(pred_xstart, num_processes)
        # rule_func = partial(FUNC_DICT[rule_name], given_key="C major")   # todo: hard code key here
        rule_func = FUNC_DICT[rule_name]
        with multiprocessing.Pool(processes=num_processes) as pool:
            gen_rule = pool.map(rule_func, pred_xstart_split)
        # Combine results
        if len(gen_rule[0].shape) == 1:    # batch_size * branching_factor < 4
            gen_rule = [item.unsqueeze(dim=0) for item in gen_rule]
        gen_rule = th.concat(gen_rule, dim=0).to(device)

    else:
        gen_rule = FUNC_DICT[rule_name](pred_xstart)
    return gen_rule


def _encode(pred_xstart, embed_model, scale_factor=1.):
    image_size_h = pred_xstart.shape[-2]
    image_size_w = pred_xstart.shape[-1]
    seq_len = image_size_w // image_size_h
    micro = th.chunk(pred_xstart, seq_len, dim=-1)  # B x C x H x W
    micro = th.concat(micro, dim=0)  # 1st second for all batch, 2nd second for all batch, ...
    micro = embed_model.encode_save(micro, range_fix=False)
    if micro.shape[1] == 8:
        z, _ = th.chunk(micro, 2, dim=1)
    else:
        z = micro
    z = th.concat(th.chunk(z, seq_len, dim=0), dim=-1)
    z = z.permute(0, 1, 3, 2)
    return z * scale_factor


def guide_schedule(t, t_start=750, t_end=0, interval=1):
    flag = t_start > t[0] >= t_end and (t[0] + 1) % interval == 0
    return flag
