"""
Compute rules from midi files. 
Could be slightly different from online compute from piano roll because saved midi is a cleaner version of piano rolls.
"""
import os
import pretty_midi
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import glob
import torch as th
import multiprocessing
from argparse import ArgumentParser
from guided_diffusion import midi_util
import torch.nn.functional as F

def main():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='loggings/cond_table/all/beam_50_1_2_cls_1',
                    help='Path to the folder that contains generated samples for rule guidance')
    parser.add_argument('--rule_name', type=str, default='chord_progression',
                    help='rule to compute from midis')
    args = parser.parse_args()

    gen_files = sorted(glob.glob(f'{args.root_dir}/*.midi') + glob.glob(f'{args.root_dir}/*.mid'))
    orig_dir = f'{args.root_dir}/gt'
    if args.rule_name is None:
        target_rules = ['pitch_hist', 'note_density', 'chord_progression']
    else:
        target_rules = [args.rule_name]

    all_results = pd.DataFrame()
    for file in tqdm(gen_files):
        gen_midi = pretty_midi.PrettyMIDI(file)
        gen_piano_roll = gen_midi.get_piano_roll(fs=100, pedal_threshold=None, onset=False)
        gen_piano_roll = th.from_numpy(gen_piano_roll)[None, None] / 63.5 - 1
        gen_piano_roll = F.pad(gen_piano_roll, (0, 1024 - gen_piano_roll.shape[3]), "constant", -1)
        basename = os.path.basename(file)
        orig_file = os.path.join(orig_dir, basename)
        try:
            orig_midi = pretty_midi.PrettyMIDI(orig_file)
        except:
            print(basename)
            continue
        orig_piano_roll = orig_midi.get_piano_roll(fs=100, pedal_threshold=None, onset=False)
        orig_piano_roll = th.from_numpy(orig_piano_roll)[None, None] / 63.5 - 1
        orig_piano_roll = F.pad(orig_piano_roll, (0, 1024 - orig_piano_roll.shape[3]), "constant", -1)

        results = midi_util.compute_rule(gen_piano_roll, orig_piano_roll, target_rules)
        all_results = pd.concat([all_results, results], ignore_index=True)

    all_results.to_csv(f'{args.root_dir}/results_computed.csv', index=False)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
