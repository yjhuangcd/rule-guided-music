# coding:utf-8
"""core_diff.py
Differentiable metrics that can be used to train VAE
Assuming piano roll is from [-1, 1], -1 is background
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def total_used_pitch(piano_roll):
    """
    total_used_pitch (Pitch count): The number of different pitches within a sample.

    Returns:
    'used_pitch': pitch count, scalar for each sample. shape: Bx1, B is batch size.
    """
    piano_roll = (piano_roll + 1) * 10    # rescale to [0,10] so that there is big enough gap between off and on
    mean_notes = torch.sum(piano_roll, dim=-1)
    used_pitch = torch.sum(torch.sigmoid(mean_notes - 10), dim=-1)   # shift 0 to -10 so that sigmoid is small
    return used_pitch


def total_pitch_class_histogram(piano_roll):
    piano_roll = (piano_roll + 1) / 2.     # rescale to [0,1]
    piano_roll = piano_roll.squeeze(dim=1)
    piano_roll_reduce_time = torch.sum(piano_roll, dim=-1)
    piano_roll_padded = torch.concat((piano_roll_reduce_time, torch.zeros(piano_roll.shape[0], 4, device=piano_roll.device)), dim=-1)
    pr_reshape = piano_roll_padded.unsqueeze(dim=1).reshape(-1, 11, 12).permute(0, 2, 1)
    histogram = pr_reshape.sum(dim=-1)
    histogram = histogram / torch.sum(histogram, dim=-1, keepdim=True)
    return histogram


def total_used_pitch_np(piano_roll):
    # testing code
    sum_notes = np.sum(piano_roll, axis=1)
    used_pitch = np.sum(sum_notes > 0)
    return used_pitch


def total_pitch_class_histogram_slow(piano_roll):
    # testing code
    """
    total_pitch_class_histogram (Pitch class histogram):
    The pitch class histogram is an octave-independent representation of the pitch content with a dimensionality of 12 for a chromatic scale.
    In our case, it represents to the octave-independent chromatic quantization of the frequency continuum.

    Returns:
    'histogram': histrogram of 12 pitch, with weighted duration shape 12
    """

    histogram = torch.zeros(12)
    for i in range(0, 128):
        pitch_class = i % 12
        histogram[pitch_class] += torch.sum(piano_roll, dim=1)[i]
    histogram = histogram / torch.sum(histogram)
    return histogram


def main():
    piano_roll_1 = np.load('/home/src/music-guided-diffusion-main/datasets/maestro-v3.0.0-overlap-fs100/test/ORIG-MIDI_03_7_8_13_Group__MID--AUDIO_19_R2_2013_wav--4_9.npy')
    # piano_roll_1 = np.load('/home/src/music-guided-diffusion-main/datasets/maestro-v3.0.0-overlap-fs100/test/ORIG-MIDI_03_7_6_13_Group__MID--AUDIO_09_R1_2013_wav--3_4.npy')
    plt.imshow(piano_roll_1[::-1])
    plt.show()
    total_used_1 = total_used_pitch_np(piano_roll_1)
    piano_roll_2 = np.load('/home/src/music-guided-diffusion-main/datasets/maestro-v3.0.0-overlap-fs100/test/ORIG-MIDI_03_7_8_13_Group__MID--AUDIO_19_R2_2013_wav--4_1.npy')
    # piano_roll_2 = np.load('/home/src/music-guided-diffusion-main/datasets/maestro-v3.0.0-overlap-fs100/test/ORIG-MIDI_03_7_6_13_Group__MID--AUDIO_09_R1_2013_wav--3_1.npy')
    plt.imshow(piano_roll_2[::-1])
    plt.show()
    total_used_2 = total_used_pitch_np(piano_roll_2)
    piano_roll_1 = torch.tensor(piano_roll_1)
    piano_roll_2 = torch.tensor(piano_roll_2)
    piano_roll = torch.stack((piano_roll_1, piano_roll_2), dim=0).unsqueeze(dim=1)
    piano_roll = piano_roll / 127 * 2 - 1   # scale to [-1, 1]
    t_pitch = total_used_pitch(piano_roll)
    print(f"gt: {total_used_1, total_used_2}, soft: {t_pitch[0,0].item()}, {t_pitch[1,0].item()}")

    piano_roll_1 = piano_roll_1 / 127
    piano_roll_2 = piano_roll_2 / 127
    hist1 = total_pitch_class_histogram_slow(piano_roll_1)
    hist2 = total_pitch_class_histogram_slow(piano_roll_2)
    hists = total_pitch_class_histogram(piano_roll)
    print("done")

if __name__ == "__main__":
    main()