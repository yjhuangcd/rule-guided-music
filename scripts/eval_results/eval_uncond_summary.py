import glob
import os
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--path_to_folder', type=str, default='loggings/eval_uncond/',
                    help='Path (absolute) to the first dataset (folder)')
args = parser.parse_args()

root_dir = args.path_to_folder
# Find all mean CSV files in subdirectories
mean_files = glob.glob(os.path.join(root_dir, '**/results_mean.csv'), recursive=True)
# Concatenate all mean CSV files
all_means = pd.concat((pd.read_csv(file) for file in mean_files), ignore_index=True)
all_means = all_means.sort_values(by=['dataset', 'method'])

# Find all std CSV files in subdirectories
std_files = glob.glob(os.path.join(root_dir, '**/results_std.csv'), recursive=True)
# Concatenate all std CSV files
all_stds = pd.concat((pd.read_csv(file) for file in std_files), ignore_index=True)
all_stds = all_stds.sort_values(by=['dataset', 'method'])

# If needed, save the concatenated DataFrames to new CSV files
all_means.to_csv(args.path_to_folder + 'summary_mean.csv', index=False)
all_stds.to_csv(args.path_to_folder + 'summary_std.csv', index=False)

print("done")
