import blobfile as bf
from tqdm import tqdm
import torch
import numpy as np
import os
import os.path as osp
import re
import torch as th
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings
import csv
import pandas as pd
from music_rule_guidance.music_rules import note_density
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (20,3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# number of excerpts, LENGTH x 1.28
LENGTH = 40
# overlap between pieces
GAP = LENGTH // 2
# mamximum empty length
ALLOWABLE_ZERO = 1  # the actual zeros would be larger than allowable because smaller 128 may have empty columns

# all dataset
SRC_DIR = 'datasets/all-128-fs100'
TGT_DIR = 'datasets/all-len-40-gap-16-no-empty'
CSV_NAME = 'datasets/all_midi.csv'
CSV_TGT_NAME = TGT_DIR
DUP_MAESTRO = True

# # pixel space
# # number of excerpts, LENGTH x 1.28
# LENGTH = 2
# # overlap between pieces
# GAP = LENGTH // 2
# # mamximum empty length
# ALLOWABLE_ZERO = 1
#
# # all dataset
# SRC_DIR = 'datasets/all-128-fs12.5'
# TGT_DIR = 'datasets/all-len-2-gap-1-pixel'
# CSV_NAME = 'datasets/all_midi.csv'
# CSV_TGT_NAME = TGT_DIR
# DUP_MAESTRO = True

# contain weird sub pieces, need to do some cutoff
MODIFY_NAMES = {'MIDI-Unprocessed_05_R1_2006_01-05_ORIG_MID--AUDIO_05_R1_2006_01_Track01_wav': 349}


def extract_number_and_string(file_name):
    ind = [i.start() for i in re.finditer('_', file_name)][-1]
    number = int(file_name[ind+1:].split('.')[0])
    name = file_name[:ind]
    return number, name


def _list_image_files(data_dir):
    dirs = bf.listdir(data_dir)
    return [data_dir + '/' + d for d in dirs]


def extract_name_from_csv(midi_filename):
    last_slash_index = midi_filename.rfind('/')
    dot_mid_index = midi_filename.rfind('.mid')
    save_name = midi_filename[last_slash_index + 1:dot_mid_index]
    return save_name


def extract_string(file_name):
    if 'loc' not in file_name:
        ind = [i.start() for i in re.finditer('_', file_name)][-1]
        name = file_name[:ind]
    else:
        ind = [i.start() for i in re.finditer('loc', file_name)][-1]
        name = file_name[:ind-1]
    return name


def find_class(name, df):
    dataset = df.loc[df['simple_midi_name'] == name]['dataset'].item()
    if dataset == 'maestro':
        result = 0
    elif dataset == 'muscore':
        result = 1
    else:
        result = 2
    return result


def main():
    if 'maestro' not in SRC_DIR:   # contains pedal and onset info
        zero_pr = np.zeros((3, 128, 128))
        prev_seq = np.zeros((3, 128, LENGTH*128))
    else:
        zero_pr = np.zeros((128, 128))
        prev_seq = np.zeros((128, LENGTH*128))
    for split in ['train', 'test']:
        target_dir = osp.join(TGT_DIR, split)
        source_dir = osp.join(SRC_DIR, split)
        if not os.path.exists(target_dir):
          os.makedirs(target_dir)
        file_names = bf.listdir(source_dir)
        name_length = defaultdict(int)
        # save length for each file
        for file in file_names:
            number, name = extract_number_and_string(file)
            if number > name_length[name]:
                name_length[name] = number

        for name in tqdm(name_length.keys()):
            if name in MODIFY_NAMES.keys():
                max_length = MODIFY_NAMES[name]
            else:
                max_length = name_length[name]+1
            # only process those that are longer than LENGTH
            if max_length >= LENGTH - ALLOWABLE_ZERO:
                first_start_inds = range(0, LENGTH, GAP)
                for first_start_ind in first_start_inds:
                    start_inds = range(first_start_ind, max_length, LENGTH)
                    for i in start_inds:
                        excerpts = []
                        zero_counts = 0
                        offset = 0   # off set because there are zeros
                        j = 0
                        last = i == start_inds[-1]
                        while j < LENGTH:
                            cur = i + j + offset
                            file_name = os.path.join(source_dir, name + '_' + str(cur) + '.npy')
                            j = j + 1
                            try:
                                excerpt = np.load(file_name)
                                excerpts.append(excerpt)
                            except:
                                # maximum searching time: LENGTH
                                if zero_counts < ALLOWABLE_ZERO or last or offset > LENGTH:
                                    # save empty piano rolls
                                    excerpt = zero_pr
                                    zero_counts += 1
                                    excerpts.append(excerpt)
                                # keep searching for non-empty rolls
                                else:
                                    j = j - 1
                                    offset = offset + 1
                        seq = np.concatenate(excerpts, axis=-1).astype(np.uint8)  # concatenate along width dimension
                        if last and zero_counts >= ALLOWABLE_ZERO:
                            sub_prev_seq = prev_seq[..., - (zero_counts - ALLOWABLE_ZERO)*128:]
                            sub_seq = seq[..., :(LENGTH - (zero_counts - ALLOWABLE_ZERO))*128]
                            seq = np.concatenate((sub_prev_seq, sub_seq), axis=-1)
                        # check emptiness of seq
                        piano_roll = th.from_numpy(seq[0][None, None])
                        piano_roll = piano_roll / 63.5 - 1
                        nd = note_density(piano_roll, quantize_factor=1, interval=128)
                        horizontal_nd = nd[LENGTH:]
                        horizontal_nd[horizontal_nd < 1e-2] = 0
                        num_nonzeros = torch.nonzero(horizontal_nd).size(0)
                        if num_nonzeros > GAP:   # more than half nonzero
                            np.save(os.path.join(target_dir, name + '_loc_' + str(first_start_ind) + '_' + str(i) + '.npy'), seq)
                        prev_seq = seq   # save for backtrack when last=True

    # write names and classes into csv
    print("write names and classes into csv...")
    for split in ['train', 'test']:
        data_dir = osp.join(TGT_DIR, split)
        if not data_dir:
            raise ValueError("unspecified data directory")
        all_files = _list_image_files(data_dir)
        # condition on 3 classes: maestro (0), muscore (1), pop (2)
        df = pd.read_csv(CSV_NAME)
        df['simple_midi_name'] = [extract_name_from_csv(midi_name) for midi_name in df['midi_filename']]
        all_file_names = bf.listdir(data_dir)
        extracted_names = [extract_string(file_name) for file_name in all_file_names]
        classes = [find_class(name, df) for name in extracted_names]

        filename = CSV_TGT_NAME + '_' + split + ".csv"
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["midi_filename", "classes"])
            csvwriter.writerows(zip(all_files, classes))

        # duplicate maestro
        if DUP_MAESTRO:
            df = pd.read_csv(filename)
            filtered_rows = df[df['classes'] == 0]
            concat_df = pd.concat([df, filtered_rows], ignore_index=True)
            concat_df.to_csv(filename, index=False)


if __name__ == "__main__":
    main()
