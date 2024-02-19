import glob
import os
import pandas as pd
from collections import defaultdict
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--root_dir', type=str, default='loggings/cond_table/single',
                    help='Path to the folder that contains generated samples for rule guidance')
parser.add_argument('--split', type=str, default='single',
                    help='Path to the folder that contains generated samples for rule guidance')
parser.add_argument('--class_label', type=int, default=1,
                    help='used class label when generation')                    
args = parser.parse_args()

# summarize each quality attributes
target_rule_names = ['pitch', 'nd', 'chord']
pattern = f'{args.root_dir}/**/quality_mean.csv'
file_paths = glob.glob(pattern, recursive=True)
class_str = f'_cls_{args.class_label}'

save_dir = os.path.join(args.root_dir, 'quality')
os.makedirs(os.path.expanduser(save_dir), exist_ok=True)

column_names = ['method', 'total_used_pitch', 'pitch_range', 'avg_IOI', 'total_pitch_class_histogram', 'mean_note_velocity', 'mean_note_duration', 'note_density']
methods = ['no_guidance', 'classifier', 'dps_nn', 'dps_rule', 'scg']

for target_rule_name in target_rule_names:
    df_mean = pd.DataFrame(columns=column_names)
    df_mean['method'] = methods
    df_mean.set_index('method', inplace=True)
    df_std = pd.DataFrame(columns=column_names)
    df_std['method'] = methods
    df_std.set_index('method', inplace=True)

    for file in file_paths:
        tmp = file.split(args.split)[1].split('/')
        extracted_target_rule = tmp[2].split(class_str)[0]
        if class_str in file and target_rule_name == extracted_target_rule:
            method = tmp[1]
            # read in mean
            df = pd.read_csv(file)
            attrs = df['attribute'].tolist()
            df.set_index('attribute', inplace=True)
            # read in std
            file_std = file.replace('mean', 'std')
            df_2 = pd.read_csv(file_std)
            df_2.set_index('attribute', inplace=True)
            for attr in attrs:
                df_mean.loc[method, attr] = df.loc[attr, 'OA'].item()
                df_std.loc[method, attr] = df_2.loc[attr, 'OA'].item()

    df_mean.reset_index(inplace=True)
    df_std.reset_index(inplace=True)

    df_mean.to_csv(os.path.join(save_dir, target_rule_name + '_quality_mean' + class_str + '.csv'), index=False)
    df_std.to_csv(os.path.join(save_dir, target_rule_name + '_quality_std' + class_str + '.csv'), index=False)

# summarize the average OA
column_names = ['method', 'pitch', 'nd', 'chord']
methods = ['no_guidance', 'classifier', 'dps_nn', 'dps_rule', 'scg']

# Create a DataFrame with 'method' as the first column
df_mean = pd.DataFrame(methods, columns=['method'])
df_std = pd.DataFrame(methods, columns=['method'])
# Reindex the DataFrame to include the other columns
df_mean = df_mean.reindex(columns=column_names)
df_std = df_std.reindex(columns=column_names)

pattern = f'{args.root_dir}/quality/*quality_mean' + class_str + '.csv'
file_paths = glob.glob(pattern, recursive=True)

for file in file_paths:
    tmp = file.split(args.split)[1].split('/')
    extracted_target_rule = tmp[2].split('_quality_mean')[0]
    if class_str in file:
        # read in mean
        df = pd.read_csv(file)
        df_mean[extracted_target_rule] = df['avg'].tolist()
        # read in std
        file_std = file.replace('mean', 'std')
        df_2 = pd.read_csv(file_std)
        df_std[extracted_target_rule] = df_2['avg'].tolist()

df_mean.to_csv(os.path.join(args.root_dir, 'quality_mean' + class_str + '.csv'), index=False)
df_std.to_csv(os.path.join(args.root_dir, 'quality_std' + class_str + '.csv'), index=False)
