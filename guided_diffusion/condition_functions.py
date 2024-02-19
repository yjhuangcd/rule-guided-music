import argparse
import os

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .pr_datasets_all import FUNC_DICT
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def model_fn(x, t, y=None, rule=None,
             model=nn.Identity(), num_classes=3, class_cond=True, cfg=False, w=0.):
    # y has to be composer, rule is a dummy input
    y_null = th.tensor([num_classes] * x.shape[0], device=x.device)
    if class_cond:
        if cfg:
            return (1 + w) * model(x, t, y) - w * model(x, t, y_null)
        else:
            return model(x, t, y)
    else:
        return model(x, t, y_null)


def dc_model_fn(x, t, y=None, rule=None,
                model=nn.Identity(), num_classes=3, class_cond=True, cfg=False, w=0.):
    # diffcollage score function takes in 4 x pitch x time
    x = x.permute(0, 1, 3, 2)
    y_null = th.tensor([num_classes] * x.shape[0], device=x.device)
    if class_cond:
        if cfg:
            eps = (1 + w) * model(x, t, y) - w * model(x, t, y_null)
            return eps.permute(0, 1, 3, 2)  # need to return 4 x time x pitch
        else:
            return model(x, t, y).permute(0, 1, 3, 2)
    else:
        return model(x, t, y_null).permute(0, 1, 3, 2)


# y is a dummy input for cond_fn, rule is the real input
def grad_nn_zt_xentropy(x, y=None, rule=None, classifier=nn.Identity()):
    # Xentropy cond_fn
    assert rule is not None
    t = th.zeros(x.shape[0], device=x.device)
    with th.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), rule.view(-1)]
        return th.autograd.grad(selected.sum(), x_in)[0]


def grad_nn_zt_mse(x, t, y=None, rule=None, classifier_scale=10., classifier=nn.Identity()):
    assert rule is not None
    with th.enable_grad():
        x_in = x.detach().requires_grad_(True)
        logits = classifier(x_in, t)
        log_probs = - F.mse_loss(logits, rule, reduction="none").sum(dim=-1)
        return th.autograd.grad(log_probs.sum(), x_in)[0] * classifier_scale


def grad_nn_zt_chord(x, t, y=None, rule=None, classifier_scale=10., classifier=nn.Identity(), both=False):
    assert rule is not None
    with th.enable_grad():
        x_in = x.detach().requires_grad_(True)
        key_logits, chord_logits = classifier(x_in, t)
        if both:
            rule_key = rule[:, :1]
            rule_chord = rule[:, 1:]
            rule_chord = rule_chord.reshape(-1)
            chord_logits = chord_logits.reshape(-1, chord_logits.shape[-1])
            key_log_probs = - F.cross_entropy(key_logits, rule_key, reduction="none")
            chord_log_probs = - F.cross_entropy(chord_logits, rule_chord, reduction="none")
            chord_log_probs = chord_log_probs.reshape(x_in.shape[0], -1).mean(dim=-1)
            log_probs = key_log_probs + chord_log_probs
        else:
            rule = rule.reshape(-1)
            chord_logits = chord_logits.reshape(-1, chord_logits.shape[-1])
            log_probs = - F.cross_entropy(chord_logits, rule, reduction="none")
        return th.autograd.grad(log_probs.sum(), x_in)[0] * classifier_scale


def nn_z0_chord_dummy(x, t, y=None, rule=None, classifier_scale=0.1, classifier=nn.Identity(), both=False):
    # classifier_scale is equivalent to step_size
    t = th.zeros(x.shape[0], device=x.device)
    key_logits, chord_logits = classifier(x, t)
    if both:
        rule_key = rule[:, :1]
        rule_chord = rule[:, 1:]
        rule_chord = rule_chord.reshape(-1)
        chord_logits = chord_logits.reshape(-1, chord_logits.shape[-1])
        key_log_probs = - F.cross_entropy(key_logits, rule_key, reduction="none")
        chord_log_probs = - F.cross_entropy(chord_logits, rule_chord, reduction="none")
        chord_log_probs = chord_log_probs.reshape(x.shape[0], -1).mean(dim=-1)
        log_probs = key_log_probs + chord_log_probs
    else:
        rule = rule.reshape(-1)
        chord_logits = chord_logits.reshape(-1, chord_logits.shape[-1])
        log_probs = - F.cross_entropy(chord_logits, rule, reduction="none")
        log_probs = log_probs.reshape(x.shape[0], -1).mean(dim=-1)
    return log_probs * classifier_scale


def nn_z0_mse_dummy(x, t, y=None, rule=None, classifier_scale=0.1, classifier=nn.Identity()):
    # mse cond_fn, t is a dummy variable b/c wrap_model in respace
    assert rule is not None
    t = th.zeros(x.shape[0], device=x.device)
    logits = classifier(x, t)
    log_probs = - F.mse_loss(logits, rule, reduction="none").sum(dim=-1)
    return log_probs * classifier_scale


def nn_z0_mse(x, rule=None, classifier=nn.Identity()):
    # mse cond_fn, t is a dummy variable b/c wrap_model in respace
    t = th.zeros(x.shape[0], device=x.device)
    logits = classifier(x, t)
    log_probs = - F.mse_loss(logits, rule, reduction="none").sum(dim=-1)
    return log_probs


def rule_x0_mse_dummy(x, t, y=None, rule=None, rule_name='pitch_hist'):
    # use differentiable rule to differentiate through rule(x_0), t is a dummy variable b/c wrap_model in respace
    logits = FUNC_DICT[rule_name](x)
    log_probs = - F.mse_loss(logits, rule, reduction="none").sum(dim=-1)
    return log_probs


def rule_x0_mse(x, rule=None, rule_name='pitch_hist', soft=False):
    # soften non-differentiable rule to differentiate through rule(x_0)
    # soften doesn't seem to work so didn't actually take in soft as input, always set to False
    logits = FUNC_DICT[rule_name](x, soft=soft)
    log_probs = - F.mse_loss(logits, rule, reduction="none").sum(dim=-1)
    return log_probs


class _WrappedFn:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x, t, y=None, rule=None):
        return self.fn(x, t, y, rule)


function_map = {
    "grad_nn_zt_xentropy": grad_nn_zt_xentropy,
    "grad_nn_zt_mse": grad_nn_zt_mse,
    "grad_nn_zt_chord": grad_nn_zt_chord,
    "nn_z0_chord_dummy": nn_z0_chord_dummy,
    "nn_z0_mse_dummy": nn_z0_mse_dummy,
    "nn_z0_mse": nn_z0_mse,
    "rule_x0_mse_dummy": rule_x0_mse_dummy,
    "rule_x0_mse": rule_x0_mse
}


def composite_nn_zt(x, t, y=None, rule=None, fns=None, classifier_scales=None, classifiers=None, rule_names=None):
    num_classifiers = len(classifiers)
    out = 0
    for i in range(num_classifiers):
        out += function_map[fns[i]](x, t, y=y, rule=rule[rule_names[i]],
                                    classifier_scale=classifier_scales[i], classifier=classifiers[i])
    return out


def composite_rule(x, t, y=None, rule=None, fns=None, classifier_scales=None, rule_names=None):
    out = 0
    for i in range(len(fns)):
        out += function_map[fns[i]](x, t, y=y, rule=rule[rule_names[i]], rule_name=rule_names[i]) * classifier_scales[i]
    return out
