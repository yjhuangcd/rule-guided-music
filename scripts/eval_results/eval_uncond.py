import glob
import pandas as pd
from collections import defaultdict
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--path_to_folder', required=True, type=str, default='loggings/eval_uncond/ours/',
                    help='Path (absolute) to the first dataset (folder)')
args = parser.parse_args()
# whether to use OA or KL
metric = 'OA'
# attribute to look at
attributes = ['total_used_pitch',
              'avg_IOI',
              'total_pitch_class_histogram',
              'pitch_range',
              'mean_note_velocity',
              'mean_note_duration',
              'note_density',
              'avg'
              ]

csv_files = glob.glob(args.path_to_folder + '*.csv')
results_mean = defaultdict(list)
results_std = defaultdict(list)

# This will print out all paths that end with '.csv'
for file_path in csv_files:
    file_name = file_path.split('/')[-1]
    dataset = file_name.split('.')[0]
    method = file_name.split('.')[1]
    stat_type = file_name.split('.')[2]
    df = pd.read_csv(file_path)
    if stat_type == 'mean':
        results_mean["dataset"].append(dataset)
        results_mean["method"].append(method)
        for attr in attributes:
            results_mean[attr].append(df[df.attribute == attr][metric].item())
    else:
        results_std["dataset"].append(dataset)
        results_std["method"].append(method)
        for attr in attributes:
            results_std[attr].append(df[df.attribute == attr][metric].item())

df_results_mean = pd.DataFrame(results_mean)
sorted_df = df_results_mean.sort_values(by=['dataset', 'method'])
sorted_df.to_csv(args.path_to_folder + 'results_mean.csv', index=False)
df_results_std = pd.DataFrame(results_std)
sorted_df = df_results_std.sort_values(by=['dataset', 'method'])
sorted_df.to_csv(args.path_to_folder + 'results_std.csv', index=False)

print("done")
