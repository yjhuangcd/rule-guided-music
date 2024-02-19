import glob
import os
import pandas as pd
from collections import defaultdict
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--root_dir', type=str, default='loggings/cond_table/single',
                    help='Path to the folder that contains generated samples for rule guidance')
parser.add_argument('--split', type=str, default='single',
                    help='Str used to split out basenames')
parser.add_argument('--class_label', type=int, default=1,
                    help='used class label when generation')                    
args = parser.parse_args()

pattern = f'{args.root_dir}/**/summary.csv'
file_paths = glob.glob(pattern, recursive=True)
class_str = f'_cls_{args.class_label}'

column_names = ['method', 'pitch_hist', 'note_density', 'chord_progression']
methods = ['no_guidance', 'classifier', 'dps_nn', 'dps_rule', 'scg']
# debug: find good weight for multi-rule
# methods = ['classifier', 'dps_nn', 'beam_50_1_2', 'beam_50_1_4', 'beam_100_1_1', 'beam_100_1_2', 'beam_100_1_2_rep', 'beam_100_1_4', 'beam_150_1_1', 'beam_200_1_1']
df_mean = pd.DataFrame(columns=column_names)
df_mean['method'] = methods
df_mean.set_index('method', inplace=True)
df_std = pd.DataFrame(columns=column_names)
df_std['method'] = methods
df_std.set_index('method', inplace=True)

for file in file_paths:
    if class_str in file:
        tmp = file.split(args.split)[1].split('/')
        method = tmp[1].split(class_str)[0]
        df = pd.read_csv(file)
        attrs = df['Attr'].tolist()
        df.set_index('Attr', inplace=True)
        for attr in attrs:
            attr_name = attr.split('.')[0]
            df_mean.loc[method, attr_name] = df.loc[attr, 'Mean'].item()
            df_std.loc[method, attr_name] = df.loc[attr, 'Std'].item()

df_mean.reset_index(inplace=True)
df_std.reset_index(inplace=True)

df_mean.to_csv(os.path.join(args.root_dir, 'loss_mean' + class_str + '.csv'), index=False)
df_std.to_csv(os.path.join(args.root_dir, 'loss_std' + class_str + '.csv'), index=False)
