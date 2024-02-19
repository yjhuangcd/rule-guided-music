import os
import math
import torch
import numpy as np
import pandas as pd
import pretty_midi
import matplotlib as mpl
import matplotlib.pyplot as plt
from . import dist_util
import yaml
from types import SimpleNamespace
from music_rule_guidance.piano_roll_to_chord import piano_roll_to_pretty_midi, KEY_DICT, IND2KEY
from music_rule_guidance.rule_maps import FUNC_DICT, LOSS_DICT
from music_rule_guidance.music_rules import MAX_PIANO, MIN_PIANO

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# bounds to compute classes for nd editing
VERTICAL_ND_BOUNDS = [1.29, 2.7578125, 3.61, 4.4921875, 5.28125, 6.1171875, 7.22]
VERTICAL_ND_CENTER = [0.56, 2.0239, 3.1839, 4.0511, 4.8867, 5.6992, 6.6686, 7.77]
HORIZONTAL_ND_BOUNDS = [1.8, 2.6, 3.2, 3.6, 4.4, 4.8, 5.8]
HORIZONTAL_ND_CENTER = [1.4, 2.2000, 2.9, 3.4, 4.0, 4.6, 5.3, 6.3]


def dict_to_obj(d):
    if isinstance(d, list):
        d = [dict_to_obj(x) if isinstance(x, dict) else x for x in d]
    if not isinstance(d, dict):
        return d
    return SimpleNamespace(**{k: dict_to_obj(v) for k, v in d.items()})


def load_config(filename):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    # Convert the dictionary to an object
    data_obj = dict_to_obj(data)
    return data_obj


@torch.no_grad()
def decode_sample_for_midi(sample, embed_model=None, scale_factor=1., threshold=-0.95):
    # decode latent samples to a long piano roll of [0, 127]
    sample = sample / scale_factor

    if embed_model is not None:
        image_size_h = sample.shape[-2]
        image_size_w = sample.shape[-1]
        if image_size_h > image_size_w:  # transposed for raster col, don't need to permute for pixel space
            sample = sample.permute(0, 1, 3, 2)  # vertical axis means pitch after transpose
        num_latents = sample.shape[-1] // sample.shape[-2]
        if image_size_h >= image_size_w:
            sample = torch.chunk(sample, num_latents, dim=-1)  # B x C x H x W
            sample = torch.concat(sample, dim=0)  # 1st second for all batch, 2nd second for all batch, ...
        sample = embed_model.decode(sample)
        if image_size_h >= image_size_w:
          sample = torch.concat(torch.chunk(sample, num_latents, dim=0), dim=-1)

    sample[sample <= threshold] = -1.  # heuristic thresholding the background
    sample = ((sample + 1) * 63.5).clamp(0, 127).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def save_piano_roll_midi(sample, save_dir, fs=100, y=None, save_piano_roll=False, save_ind=0):
    # input shape: B x 128 (pitch) x time (no pedal) or B x 2 (pedal) x 128 x time (with pedal)
    fig_size = sample.shape[-1] // 128 * 3
    plt.rcParams["figure.figsize"] = (fig_size, 3)
    pedal = True if len(sample.shape) == 4 else False
    onset = True if sample.shape[1] == 3 else False
    for i in range(sample.shape[0]):
        cur_sample = sample[i]
        if cur_sample.shape[-1] < 5000 and save_piano_roll:  # do not save piano rolls that are too long
          if pedal:
            plt.imshow(cur_sample[0, ::-1], vmin=0, vmax=127)
          else:
            plt.imshow(cur_sample[::-1], vmin=0, vmax=127)
          plt.savefig(os.path.join(save_dir, "prsample_" + str(i) + ".png"))
        if onset:
          # add onset for first column
          first_column = cur_sample[0, :, 0]
          first_onset_pitch = first_column.nonzero()[0]
          cur_sample[1, first_onset_pitch, 0] = 127
        cur_sample = cur_sample.astype(np.float32)
        pm = piano_roll_to_pretty_midi(cur_sample, fs=fs)
        if y is not None:
            save_name = 'sample_' + str(i + save_ind) + '_y_' + str(y[i].item()) + '.midi'
        else:
            save_name = 'sample_' + str(i + save_ind) + '.midi'
        pm.write(os.path.join(save_dir, save_name))
    return


def eval_rule_loss(generated_samples, target_rules):
    results = {}
    batch_size = generated_samples.shape[0]
    for rule_name, rule_target in target_rules.items():
        rule_target_list = rule_target.tolist()
        if batch_size == 1:
            rule_target_list = [rule_target_list]
        results[rule_name + '.target_rule'] = rule_target_list
        rule_target = rule_target.to(generated_samples.device)
        if 'chord' in rule_name:
            gen_rule, key, corr = FUNC_DICT[rule_name](generated_samples, return_key=True)
            key_strings = [IND2KEY[key_ind] for key_ind in key]
            loss = LOSS_DICT[rule_name](gen_rule, rule_target)
            mean_loss, std_loss, gen_rule, loss = loss.mean(), loss.std(), gen_rule.tolist(), loss.tolist()
            if batch_size == 1:
                gen_rule = [gen_rule]
            results[rule_name + '.gen_rule'] = gen_rule
            results[rule_name + '.key_str'] = key_strings
            results[rule_name + '.key_corr'] = corr
            results[rule_name + '.loss'] = loss
        else:
            gen_rule = FUNC_DICT[rule_name](generated_samples)
            loss = LOSS_DICT[rule_name](gen_rule, rule_target)
            mean_loss, std_loss, gen_rule, loss = loss.mean(), loss.std(), gen_rule.tolist(), loss.tolist()
            if batch_size == 1:
                gen_rule = [gen_rule]
            results[rule_name + '.gen_rule'] = gen_rule
            results[rule_name + '.loss'] = loss
    return pd.DataFrame(results)


def compute_rule(generated_samples, orig_samples, target_rules):
    results = {}
    batch_size = generated_samples.shape[0]
    for rule_name in target_rules:
        rule_target = FUNC_DICT[rule_name](orig_samples)
        rule_target_list = rule_target.tolist()
        if batch_size == 1:
            rule_target_list = [rule_target_list]
        results[rule_name + '.target_rule'] = rule_target_list
        rule_target = rule_target.to(generated_samples.device)
        if rule_name == 'chord_progression':
            gen_rule, key, corr = FUNC_DICT[rule_name](generated_samples, return_key=True)
            key_strings = [IND2KEY[key_ind] for key_ind in key]
            loss = LOSS_DICT[rule_name](gen_rule, rule_target)
            mean_loss, std_loss, gen_rule, loss = loss.mean(), loss.std(), gen_rule.tolist(), loss.tolist()
            if batch_size == 1:
                gen_rule = [gen_rule]
            results[rule_name + '.gen_rule'] = gen_rule
            results[rule_name + '.key_str'] = key_strings
            results[rule_name + '.key_corr'] = corr
            results[rule_name + '.loss'] = loss
        else:
            gen_rule = FUNC_DICT[rule_name](generated_samples)
            loss = LOSS_DICT[rule_name](gen_rule, rule_target)
            mean_loss, std_loss, gen_rule, loss = loss.mean(), loss.std(), gen_rule.tolist(), loss.tolist()
            if batch_size == 1:
                gen_rule = [gen_rule]
            results[rule_name + '.gen_rule'] = gen_rule
            results[rule_name + '.loss'] = loss
    return pd.DataFrame(results)


def visualize_piano_roll(piano_roll):
    """
    Assuming piano roll has shape Bx1x128x1024, and the values are between [-1, 1], on gpu.
    Visualize with some gap in between the first 256, last 256/
    """
    piano_roll = torch.flip(piano_roll, [2])
    piano_roll = (piano_roll + 1) / 2.
    vis_length = 256
    gap = 80
    plt.rcParams["figure.figsize"] = (12, 3)
    data = torch.zeros(128, vis_length * 2 + gap)
    data[:, :vis_length] = piano_roll[0, 0, :, :vis_length]
    data[:, -vis_length:] = piano_roll[0, 0, :, -vis_length:]
    data_clone = data.clone()
    # make it look thicker
    data[1:, :] = data[1:, :] + data_clone[:-1, :]
    data[2:, :] = data[2:, :] + data_clone[:-2, :]
    data = data.cpu().numpy()
    plt.imshow(data, cmap=mpl.colormaps['Blues'])
    ax = plt.gca()  # gca stands for 'get current axis'
    for edge, spine in ax.spines.items():
      spine.set_linewidth(2)  # Adjust the value as per your requirement
    plt.grid(color='gray', linestyle='-', linewidth=2., alpha=0.5, which='both', axis='x')
    plt.xticks(
      np.concatenate((np.arange(0, vis_length + 1, 128), np.arange(vis_length + gap, vis_length * 2 + gap, 128))))
    # plt.savefig('piano_roll_example.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
    plt.tight_layout()
    plt.show()

    plt.rcParams["figure.figsize"] = (3, 3)
    for i in range(2):
      plt.imshow(data[:, i*128: (i+1)*128], cmap=mpl.colormaps['Blues'])
      ax = plt.gca()
      for edge, spine in ax.spines.items():
        spine.set_linewidth(2)
      plt.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
      plt.tight_layout()
      plt.show()

    for i in range(-2, 0):
      if (i+1)*128 < 0:
        plt.imshow(data[:, i*128: (i+1)*128], cmap=mpl.colormaps['Blues'])
      else:
        plt.imshow(data[:, i*128:], cmap=mpl.colormaps['Blues'])
      ax = plt.gca()
      for edge, spine in ax.spines.items():
        spine.set_linewidth(2)
      plt.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
      plt.tight_layout()
      plt.show()

    return


def visualize_full_piano_roll(midi_file_name, fs=100):
    """
    Visualize full piano roll from midi file
    """
    midi_data = pretty_midi.PrettyMIDI(midi_file_name)
    # do not process sustain pedal
    piano_roll = torch.tensor(midi_data.get_piano_roll(fs=fs, pedal_threshold=None))
    data = torch.flip(piano_roll, [0])
    plt.rcParams["figure.figsize"] = (12, 3)
    # data_clone = data.clone()
    # # make it look thicker
    # data[1:, :] = data[1:, :] + data_clone[:-1, :]
    # data[2:, :] = data[2:, :] + data_clone[:-2, :]
    data = data.cpu().numpy()
    plt.imshow(data, cmap=mpl.colormaps['Blues'])
    ax = plt.gca()  # gca stands for 'get current axis'
    for edge, spine in ax.spines.items():
      spine.set_linewidth(2)  # Adjust the value as per your requirement
    plt.grid(color='gray', linestyle='-', linewidth=2., alpha=0.5, which='both', axis='x')
    plt.xticks(np.arange(0, piano_roll.shape[1], 128))
    # plt.savefig('piano_roll_example.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
    plt.tight_layout()
    plt.show()
    return


def plot_record(vals, title, save_dir):
    ts = [item[0] for item in vals]
    log_probs = [item[1] for item in vals]
    plt.plot(ts, log_probs)
    plt.gca().invert_xaxis()
    plt.title(title)
    plt.savefig(save_dir + '/' + title + '.png')
    plt.show()
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
