# Modified from: https://github.com/RichardYang40148/mgeval/blob/master/__main__.py

from argparse import ArgumentParser
import glob
import copy
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import pretty_midi
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
import random

from mgeval import core, utils

def delete_nan(arr):
    arr[np.isnan(arr) | np.isinf(arr)] = 0
    return arr


parser = ArgumentParser()
parser.add_argument('--set1dir', required=True, type=str,
                    help='Path (absolute) to the first dataset (folder)')
parser.add_argument('--set2dir', required=True, type=str,
                    help='Path (absolute) to the second dataset (folder)')
parser.add_argument('--outdir', required=True, type=str,
                    help='Directory where the figures and analysis will be stored')
parser.add_argument('--savename', type=str, default=None,
                    help='savename of the output csv')                    
parser.add_argument('--num_sample', required=False, type=int,
                    help='Number of Samples to be evaluated')
parser.add_argument('--num_runs', required=False, type=int,
                    help='Number of Runs to compute mean and std')


args = parser.parse_args()
print(args.set1dir)
print(args.set2dir)

set1_all = glob.glob(os.path.join(args.set1dir, '*.midi')) + glob.glob(os.path.join(args.set1dir, '*.mid'))
set2_all = glob.glob(os.path.join(args.set2dir, '*.midi')) + glob.glob(os.path.join(args.set2dir, '*.mid'))
set2 = random.sample(set2_all, min(args.num_sample, len(set2_all)))
if not any(set2):
    print("Error: baseline set it empty")
    exit()

current_path = os.getcwd()
save_dir = os.path.join(current_path, args.outdir)
os.makedirs(os.path.expanduser(save_dir), exist_ok=True)


print('Evaluation begins ~')

dfs = []
for _ in range(args.num_runs):
    # select random part in the gt dataset
    set1 = random.sample(set1_all, min(args.num_sample, len(set1_all)))
    if not any(set1):
      print("Error: sample set it empty")
      exit()

    # Initialize Evaluation Set
    if args.num_sample:
        num_samples = min(args.num_sample, min(len(set2), len(set1)))
    else:
        num_samples = min(len(set2), len(set1))
    # num_samples = 10 # small number for faster testing and debugging purpose

    print("Number of samples in use: ", num_samples)
    evalset = {
                'total_used_pitch': np.zeros((num_samples, 1))
              , 'pitch_range': np.zeros((num_samples, 1))
              , 'avg_IOI': np.zeros((num_samples, 1))
              , 'total_pitch_class_histogram': np.zeros((num_samples, 12))
              , 'mean_note_velocity':np.zeros((num_samples, 1))
              , 'mean_note_duration':np.zeros((num_samples, 1))
              , 'note_density':np.zeros((num_samples, 1))
              }


    # print(evalset)

    metrics_list = list(evalset.keys())

    single_arg_metrics = (
        [ 'total_used_pitch'
        , 'avg_IOI'
        , 'total_pitch_class_histogram'
        , 'pitch_range'
        , 'mean_note_velocity'
        , 'mean_note_duration'
        , 'note_density'
        , 'pitch_class_transition_matrix'
        ])

    set1_eval = copy.deepcopy(evalset)
    set2_eval = copy.deepcopy(evalset)

    sets = [ (set1, set1_eval), (set2, set2_eval) ]


    # Extract Features
    for _set, _set_eval in sets:
        for i in range(0, num_samples):
            feature = core.extract_feature(_set[i])
            for metric in metrics_list:
                evaluator = getattr(core.metrics(), metric)
                if metric in single_arg_metrics:
                    tmp = evaluator(feature)
                else:
                    tmp = evaluator(feature, 0)
                _set_eval[metric][i] = tmp

    loo = LeaveOneOut()
    loo.get_n_splits(np.arange(num_samples))
    set1_intra = np.zeros((num_samples, len(metrics_list), num_samples - 1))
    set2_intra = np.zeros((num_samples, len(metrics_list), num_samples - 1))


    # Calculate Intra-set Metrics
    for i, metric in enumerate(metrics_list):
        for train_index, test_index in loo.split(np.arange(num_samples)):
            set1_intra[test_index[0]][i] = utils.c_dist(
                set1_eval[metrics_list[i]][test_index], set1_eval[metrics_list[i]][train_index])
            set2_intra[test_index[0]][i] = utils.c_dist(
                set2_eval[metrics_list[i]][test_index], set2_eval[metrics_list[i]][train_index])

    loo = LeaveOneOut()
    loo.get_n_splits(np.arange(num_samples))
    sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))

    # Calculate Inter-set Metrics
    for i, metric in enumerate(metrics_list):
        for train_index, test_index in loo.split(np.arange(num_samples)):
            sets_inter[test_index[0]][i] = utils.c_dist(set1_eval[metric][test_index], set2_eval[metric])


    plot_set1_intra = np.transpose(
        set1_intra, (1, 0, 2)).reshape(len(metrics_list), -1)
    plot_set2_intra = np.transpose(
        set2_intra, (1, 0, 2)).reshape(len(metrics_list), -1)
    plot_sets_inter = np.transpose(
        sets_inter, (1, 0, 2)).reshape(len(metrics_list), -1)

    plot_set1_intra = delete_nan(plot_set1_intra)
    plot_set2_intra = delete_nan(plot_set2_intra)
    plot_sets_inter = delete_nan(plot_sets_inter)

    # output = {}
    df_output = defaultdict(list)
    for i, metric in enumerate(metrics_list):
        print("-----------------------------")
        print('calculating KL and OA of: {}'.format(metric))

        mean = np.mean(set1_eval[metric], axis=0).tolist()
        std = np.std(set1_eval[metric], axis=0).tolist()
        filename = metric+".png"

        plt.figure()
        sns.kdeplot(plot_set1_intra[i], label='intra_set1')
        sns.kdeplot(plot_sets_inter[i], label='inter')
        sns.kdeplot(plot_set2_intra[i], label='intra_set2')

        plt.title(metrics_list[i])
        plt.xlabel('Euclidean distance')
        plt.legend()
        # plt.savefig(f'{save_dir}/{filename}')
        plt.show()
        plt.close()

        kl1 = utils.kl_dist(plot_set1_intra[i], plot_sets_inter[i])
        ol1 = utils.overlap_area(plot_set1_intra[i], plot_sets_inter[i])
        # kl2 = utils.kl_dist(plot_set2_intra[i], plot_sets_inter[i])
        # ol2 = utils.overlap_area(plot_set2_intra[i], plot_sets_inter[i])

        print("KL(set1 || inter_set): ", kl1)
        print("OA(set1, inter1): ", ol1)
        # print("KL(set2 || inter_set): ", kl2)
        # print("OA(set2, inter1): ", ol2)
        # output[metric] = [mean, std, kl1, ol1, kl2, ol2]
        df_output['attribute'].append(metric)
        df_output['KL'].append(kl1)
        df_output['OA'].append(ol1)
    df = pd.DataFrame(df_output)
    avg = {'attribute': 'avg', 'KL': df['KL'].mean(), 'OA': df['OA'].mean()}
    avg_df = pd.DataFrame([avg])
    df = pd.concat([df, avg_df], ignore_index=True)
    dfs.append(df)

# Concatenate the values of the 'KL' and 'OA' columns across all dataframes
kl_values = np.column_stack([df['KL'] for df in dfs])
oa_values = np.column_stack([df['OA'] for df in dfs])

# Calculate the mean and standard deviation for 'KL' and 'OA' columns
kl_mean = np.mean(kl_values, axis=1)
kl_std = np.std(kl_values, axis=1)
oa_mean = np.mean(oa_values, axis=1)
oa_std = np.std(oa_values, axis=1)

# Create new dataframes for mean and standard deviation
mean_df = pd.DataFrame({'attribute': df['attribute'], 'KL': kl_mean, 'OA': oa_mean})
std_df = pd.DataFrame({'attribute': df['attribute'], 'KL': kl_std, 'OA': oa_std})

# save statistics into a .txt file for reading
if args.savename is None:
    dataset = args.set1dir.split('/')[-1]
    model = args.set2dir.split('/')[-1]
    mean_output_csv = f'{save_dir}/{dataset}.{model}.mean.csv'
    std_output_csv = f'{save_dir}/{dataset}.{model}.std.csv'
else:
    mean_output_csv = f'{save_dir}/{args.savename}_mean.csv'
    std_output_csv = f'{save_dir}/{args.savename}_std.csv'

mean_df.to_csv(mean_output_csv, index=False)
std_df.to_csv(std_output_csv, index=False)

print('Saved output to {} folder.'.format(args.outdir))
print('Evaluation Complete.')