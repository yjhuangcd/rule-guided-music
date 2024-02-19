import math
import random
import os
import pandas as pd
import csv
import re
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from music_rule_guidance import music_rules
from music_rule_guidance.rule_maps import FUNC_DICT

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (6,3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# This file load in merged dataset with y being its dataset info


def load_data(
    *,
    data_dir,
    batch_size,
    class_cond=False,
    deterministic=False,
    image_size=1024,
    rule=None,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: the csv file that contains all the data paths and classes.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param rule: a str that contains the name of the rule
    """

    df = pd.read_csv(data_dir)
    all_files = df['midi_filename'].tolist()
    classes = None
    if class_cond:
        classes = df['classes'].tolist()
    if deterministic:
        dataset = ImageDataset(
            all_files,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            image_size=image_size,
            rule=rule,
            pitch_shift=False,
            time_stretch=False,
        )
    else:
        dataset = ImageDataset(
            all_files,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            image_size=image_size,
            rule=rule,
        )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def key_shift(x, k):
    # apply shift on both notes and onset
    # x     sample (batch x 3 x pitch x time)
    # k     number of pitches to shift
    # only apply on (batch x 2 x pitch x time) because no key shift on pedal

    pitches_and_onsets = x[:, :2, :, :]
    pedals = x[:, 2:, :, :]

    if k > 0:
        pitches_and_onsets = torch.cat((pitches_and_onsets[:, :, k:, :], pitches_and_onsets[:, :, 0:k, :]), dim=2)
    elif k < 0:
        pitches_and_onsets = torch.cat((pitches_and_onsets[:, :, -k:, :], pitches_and_onsets[:, :, 0:-k, :]), dim=2)

    x = torch.cat((pitches_and_onsets, pedals), dim=1)
    return music_rules.piano_like(x)


class ImageDataset(Dataset):
    def __init__(
        self,
        image_paths,
        classes=None,
        rule=None,
        shard=0,
        num_shards=1,
        image_size=1024,
        pitch_shift=True,
        time_stretch=True,
    ):
        super().__init__()
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.rule = rule
        self.pitch_shift = pitch_shift
        self.time_stretch = time_stretch
        self.image_size = image_size

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        arr = np.load(path)[np.newaxis]   # 1 x 2 x 128 x time
        arr = arr.astype(np.float32) / 63.5 - 1
        arr = torch.from_numpy(arr)

        if self.time_stretch:   # apply for both notes and pedal
            pr_len = int(np.random.uniform(0.95, 1.05) * self.image_size)
            start = np.random.randint(arr.shape[-1] - pr_len)
            arr = arr[:, :, :, start:start+pr_len]
            if pr_len < self.image_size:   # stretching, prevent duplicating onsets
                piano_pedal = arr[:, [0, 2], :, :]
                piano_pedal = F.interpolate(piano_pedal, size=(128, self.image_size), mode='nearest')
                onset_raw = arr[:, 1:2, :, :]
                ind_a2b = (torch.arange(self.image_size)/self.image_size*pr_len).int()
                ind = ind_a2b.diff().nonzero().squeeze() + 1
                zero_tensor = torch.tensor([0])
                ind = torch.concat((zero_tensor, ind))
                onset = -torch.ones(1, 1, 128, self.image_size)
                onset[:, :, :, ind] = onset_raw
                arr = torch.concat((piano_pedal[:, :1, :, :], onset, piano_pedal[:, 1:, :, :]), dim=1)
            if pr_len > self.image_size:  # compressing, add onset if happen to drop onsets and keep durations
                arr = F.interpolate(arr, size=(128, self.image_size), mode='nearest')
                piano = arr[:, :1, :, :]
                first_column = piano[:, :, :, :1]
                padded_piano = torch.concat((first_column, piano), dim=-1)
                onset_online = torch.diff(padded_piano, dim=-1)
                mask = onset_online > 0
                arr[:, 1:2, :, :][mask] = 1
        else:
            arr = arr[:, :, :, :self.image_size]
        if self.pitch_shift:   # only apply for notes
            k = np.random.randint(-6, 7)   # generate randint from -6 to +6
            arr = key_shift(arr, k)

        arr = music_rules.piano_like(arr)   # also set pedal roll to be 0 for non-piano pitches (match VAE training)

        out_dict = {}
        if self.rule is not None:
            if 'chord' in self.rule:  # predict chord and key jointly
              chord, key, _ = FUNC_DICT[self.rule](arr, return_key=True)
              out_dict["chord"] = chord
              out_dict["key"] = np.array(key, dtype=np.int64)
            else:
              out_dict[self.rule] = FUNC_DICT[self.rule](arr)
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        # debug
        # out_dict["path"] = path
        # Remove the extra dimensions to get back a 3D tensor: 2x128x128
        arr = arr.squeeze(0)
        return arr, out_dict

