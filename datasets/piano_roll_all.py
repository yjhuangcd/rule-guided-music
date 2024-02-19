import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pretty_midi
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
from music_rule_guidance.music_rules import MAX_PIANO, MIN_PIANO

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (6,3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

CC_SUSTAIN_PEDAL = 64


def split_csv(csv_path='merged_midi.csv'):
  # separate training validation testing files
  df = pd.read_csv(csv_path)
  save_name = csv_path[:csv_path.rfind('.csv')]
  for split in ['train', 'validation', 'test']:
    path = os.path.join(save_name, split + '.csv')
    df_sub = df[df.split == split]
    df_sub.to_csv(path, index=False)
  return


def quantize_pedal(value, num_bins=8):
  """Quantize an integer value from 0 to 127 into 8 bins and return the center value of the bin."""
  if value < 0 or value > 127:
    raise ValueError("Value should be between 0 and 127")
  # Determine bin size
  bin_size = 128 // num_bins  # 16
  # Quantize the value
  bin_index = value // bin_size
  bin_center = bin_size * bin_index + bin_size // 2
  # Handle edge case for the last bin
  if bin_center > 127:
    bin_center = 127
  return bin_center


def get_full_piano_roll(midi_data, fs, show=False):
  # do not process sustain pedal
  piano_roll, onset_roll = midi_data.get_piano_roll(fs=fs, pedal_threshold=None, onset=True)
  # save pedal roll explicitly
  pedal_roll = np.zeros_like(piano_roll)
  # process pedal
  for instru in midi_data.instruments:
    pedal_changes = [_e for _e in instru.control_changes if _e.number == CC_SUSTAIN_PEDAL]
    for cc in pedal_changes:
      time_now = int(cc.time * fs)
      if time_now < pedal_roll.shape[-1]:
        # need to distinguish control_change 0 and background 0, with quantize 0-16 will be 8
        # in muscore files, 0 immediately followed by 127, need to shift by one column
        if pedal_roll[MIN_PIANO, time_now] != 0. and abs(pedal_roll[MIN_PIANO, time_now] - cc.value) > 64:
          # use shift 2 here to prevent missing change when using interpolation augmentation
          pedal_roll[MIN_PIANO:MAX_PIANO + 1, min(time_now + 2, pedal_roll.shape[-1] - 1)] = quantize_pedal(cc.value)
        else:
          pedal_roll[MIN_PIANO:MAX_PIANO + 1, time_now] = quantize_pedal(cc.value)
  full_roll = np.concatenate((piano_roll[None], onset_roll[None], pedal_roll[None]), axis=0)
  if show:
    plt.imshow(piano_roll[::-1, :1024], vmin=0, vmax=127)
    plt.show()
    plt.imshow(pedal_roll[::-1, :1024], vmin=0, vmax=127)
    plt.show()
  return full_roll


def preprocess_midi(target='merged', csv_path='merged_midi.csv', fs=100., image_size=128, overlap=False, show=False):
  # get piano roll from midi file
  df = pd.read_csv(csv_path)
  total_pieces = len(df)
  if not os.path.exists(target):
    os.makedirs(target)
  for split in ['train', 'test']:
    path = os.path.join(target, split)
    if not os.path.exists(path):
      os.makedirs(path)
  for i in tqdm(range(total_pieces)):
    midi_filename = df.midi_filename[i]
    split = df.split[i]
    dataset = df.dataset[i]
    path = os.path.join(target, split)
    midi_data = pretty_midi.PrettyMIDI(os.path.join(dataset, midi_filename))
    full_roll = get_full_piano_roll(midi_data, fs=fs, show=show)
    for j in range(0, full_roll.shape[-1], image_size):
      if j + image_size <= full_roll.shape[-1]:
        full_roll_excerpt = full_roll[:, :, j:j + image_size]
      else:
        full_roll_excerpt = np.zeros((3, full_roll.shape[1], image_size))   # 2x128ximage_size
        full_roll_excerpt[:, :, : full_roll.shape[-1] - j] = full_roll[:, :, j:]
      empty_roll = math.isclose(full_roll_excerpt.max(), 0.)
      if not empty_roll:
        # Find the last '/' in the string
        last_slash_index = midi_filename.rfind('/')
        # Find the '.npy' in the string
        dot_mid_index = midi_filename.rfind('.mid')
        # Extract the substring between last '/' and '.mid'
        save_name = midi_filename[last_slash_index + 1:dot_mid_index]
        full_roll_excerpt = full_roll_excerpt.astype(np.uint8)
        np.save(os.path.join(path, save_name + '_' + str(j // image_size) + '.npy'), full_roll_excerpt)
        # save with dataset name for VAE duplicate file names
        # np.save(os.path.join(path, dataset + '_' + save_name + '_' + str(j // image_size) + '.npy'), full_roll_excerpt)
    if overlap:
      for j in range(image_size//2, full_roll.shape[-1], image_size):   # overlap with image_size//2
        if j + image_size <= full_roll.shape[-1]:
          full_roll_excerpt = full_roll[:, :, j:j + image_size]
        else:
          full_roll_excerpt = np.zeros((3, full_roll.shape[1], image_size))
          full_roll_excerpt[:, :, : full_roll.shape[-1] - j] = full_roll[:, :, j:]
        empty_roll = math.isclose(full_roll_excerpt.max(), 0.)
        if not empty_roll:
          last_slash_index = midi_filename.rfind('/')
          dot_mid_index = midi_filename.rfind('.mid')
          save_name = midi_filename[last_slash_index + 1:dot_mid_index]
          full_roll_excerpt = full_roll_excerpt.astype(np.uint8)
          np.save(os.path.join(path, 'shift_' + save_name + '_' + str(j // image_size) + '.npy'), full_roll_excerpt)
          # save with dataset name for VAE duplicate file names
          # np.save(os.path.join(path, dataset + '_' + 'shift_' + save_name + '_' + str(j // image_size) + '.npy'), full_roll_excerpt)
  return


def main():
    # create fs=100 1.28s datasets without overlap (can be rearranged)
    preprocess_midi(target='all-128-fs100', csv_path='all_midi.csv', fs=100, image_size=128, overlap=False, show=False)
    # create fs=100 2.56s datasets with overlap (used for vae training), when load in, need to select 1.28s from 2.56s
    # preprocess_midi(target='all-256-overlap-fs100', csv_path='all_midi.csv', fs=100, image_size=256, overlap=True,
    #                 show=False)
    # create fs=12.5 (0.08s) for pixel space diffusion model, rearrangement with length 2
    # preprocess_midi(target='all-128-fs12.5', csv_path='all_midi.csv', fs=12.5, image_size=128, overlap=False,
    #                 show=False)


if __name__ == "__main__":
    main()