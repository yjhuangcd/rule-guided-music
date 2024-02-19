# coding:utf-8
"""core_diff.py
Differentiable metrics that can be used to train VAE
Assuming piano roll is from [-1, 1], -1 is background
Input size from data loader: 1x1x128xLENGTH
"""

import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from .piano_roll_to_chord import piano_roll_to_chords, piano_roll_to_chords_save_midi

# bounds to compute classes for nd editing
VERTICAL_ND_BOUNDS = [1.29, 2.7578125, 3.61, 4.4921875, 5.28125, 6.1171875, 7.22]
VERTICAL_ND_CENTER = [0.56, 2.0239, 3.1839, 4.0511, 4.8867, 5.6992, 6.6686, 7.77]
HORIZONTAL_ND_BOUNDS = [1.8, 2.6, 3.2, 3.6, 4.4, 4.8, 5.8]
HORIZONTAL_ND_CENTER = [1.4, 2.2000, 2.9, 3.4, 4.0, 4.6, 5.3, 6.3]
MIN_PIANO, MAX_PIANO, OFF = 21, 108, -1


def piano_like(x):
  x[:, :, :MIN_PIANO, :] = OFF
  x[:, :, MAX_PIANO + 1:, :] = OFF
  return x


def total_pitch_class_histogram(piano_roll):
    # only take the first channel of notes (ignore pedal)
    piano_roll = piano_roll[:, :1, :, :]
    piano_roll = piano_like(piano_roll)
    piano_roll = (piano_roll + 1) / 2.     # rescale to [0,1]
    piano_roll = piano_roll.squeeze(dim=1)
    piano_roll_reduce_time = torch.sum(piano_roll, dim=-1)
    piano_roll_padded = torch.concat((piano_roll_reduce_time, torch.zeros(piano_roll.shape[0], 4, device=piano_roll.device)), dim=-1)
    pr_reshape = piano_roll_padded.unsqueeze(dim=1).reshape(-1, 11, 12).permute(0, 2, 1)
    histogram = pr_reshape.sum(dim=-1)
    histogram = histogram / (torch.sum(histogram, dim=-1, keepdim=True) + 1e-12)   # 1e-12 to prevent from dividing by 0
    if histogram.shape[0] == 1:
        return histogram.squeeze(dim=0)   # output size: 12
    else:
        return histogram


def note_density(piano_roll, interval=128, quantize_factor=1, horizontal_scale=5):
    # default hr_scale=5, set horizontal scale=2 for pop
    # todo: set quantize=4 for demo
    """
    return both vertical and horizontal note density
    vertical density is the average number of vertical notes per column every interval
    horizontal density is the total number of horizontal notes every interval / 5
    quantize_factor: tolerant to slight mismatch in time
    horizontal_scale: rescale horizontal nd so it's on the same order of mag as vertical nd
    """
    piano_roll = piano_roll[:, :1, :, :]
    batch_size = piano_roll.shape[0]
    orig_size = piano_roll.shape[-1]
    if quantize_factor != 1:
        piano_roll = F.interpolate(piano_roll, size=(128, orig_size // quantize_factor), mode='nearest')
        interval = interval // quantize_factor
    piano_roll = piano_like(piano_roll)

    # thresholding because we need to find nonzero notes
    piano_roll[piano_roll < -0.95] = -1.
    piano_roll = (piano_roll + 1) / 2.  # rescale to [0,1] and has shape Bx1x128xLENGTH
    # make it binary to count number of notes
    piano_roll[piano_roll >= 1e-2] = 1.
    piano_roll[piano_roll < 1e-2] = 0.
    vertical_nd_per_col = piano_roll.sum(dim=2)   # B, 1, LENGTH
    piano_roll = F.pad(piano_roll, (1, 1), 'constant')
    diff_piano_roll = torch.diff(piano_roll)
    diff_piano_roll[diff_piano_roll < 0] = 0
    horizontal_nd_per_col = diff_piano_roll.sum(dim=2)[:, :, :-1]    # B, 1, LENGTH
    horizontal_nd_per_col[horizontal_nd_per_col != 0.] = 1
    vertical_nd = vertical_nd_per_col.reshape(batch_size, 1, -1, interval).mean(dim=-1)
    horizontal_nd = horizontal_nd_per_col.reshape(batch_size, 1, -1, interval).sum(dim=-1) / horizontal_scale
    # concat as a label for training classifiers
    nd = torch.concat((vertical_nd, horizontal_nd), dim=-1)
    if batch_size == 1:
        return nd.squeeze()
    else:
        return nd.squeeze(dim=1)


def note_density_class(piano_roll, interval=128, quantize_factor=1, horizontal_scale=1):
    vt_bounds = torch.tensor(VERTICAL_ND_BOUNDS).to(piano_roll.device)
    hr_bounds = torch.tensor(HORIZONTAL_ND_BOUNDS).to(piano_roll.device) / horizontal_scale
    orig_rule = note_density(piano_roll, interval=interval, quantize_factor=quantize_factor, horizontal_scale=horizontal_scale)
    total_length = orig_rule.shape[-1]
    vt_nd_classes = torch.bucketize(orig_rule[:, :total_length // 2], vt_bounds)
    hr_nd_classes = torch.bucketize(orig_rule[:, total_length // 2:], hr_bounds)
    target_rule = torch.concat((vt_nd_classes, hr_nd_classes), dim=-1)
    return target_rule


def get_chords(piano_roll_batch, given_key=None, fs=100, window_size=1.28, return_key=False):
    # piano_roll_batch: Bx1x128x1024
    # given_key: assuming the key is given, compute the chords based on the given key
    piano_roll_batch = piano_roll_batch[:, :1, :, :]
    if not return_key:
        out_all = []
    else:
        out_chord_all = []
        out_key_all = []
        out_key_corr_all = []
    piano_roll_batch = piano_like(piano_roll_batch)
    piano_roll_batch[piano_roll_batch < -0.95] = -1.
    piano_roll_batch = (piano_roll_batch + 1) / 2 * 127
    piano_roll_batch = torch.clamp(piano_roll_batch, min=0, max=127)
    for i in range(piano_roll_batch.shape[0]):
        piano_roll = piano_roll_batch[i, 0].cpu().numpy().astype(np.intc)
        # todo: pass in given key and window_size
        out = piano_roll_to_chords(piano_roll, given_key=given_key, fs=fs, window_size=window_size, return_key=return_key)
        if return_key:
            out_chord_all.append(out["chords"].unsqueeze(dim=0))
            out_key_all.append(out["key"])
            out_key_corr_all.append(out["correlationCoefficient"])
        else:
            out_all.append(out["chords"].unsqueeze(dim=0))
    if return_key:
        chords = torch.concat(out_chord_all, dim=0)
        if chords.shape[0] == 1:
            chords = chords.squeeze(dim=0)
        return chords, out_key_all, out_key_corr_all
    else:
        chords = torch.concat(out_all, dim=0)
        if chords.shape[0] == 1:
            chords = chords.squeeze(dim=0)
        return chords
