import pandas as pd
import numpy as np
import ast
from collections import defaultdict
from argparse import ArgumentParser
# Bins computed from gt samples
from guided_diffusion.midi_util import VERTICAL_ND_BOUNDS, HORIZONTAL_ND_BOUNDS

parser = ArgumentParser()
parser.add_argument('--file_name', type=str, default='loggings/edit_table/nd_500_cls_2/results.csv',
                    help='Path to the folder that contains generated samples for rule guidance')
parser.add_argument('--rule_name', type=str, default='note_density',
                    help='Path to the folder that contains generated samples for rule guidance') 
parser.add_argument('--horizontal_scale', type=float, default=1.,
                    help='scale horizontal note density')                                       
parser.add_argument('--bins', type=int, default=8,
                    help='Str used to split out basenames')
args = parser.parse_args()


HORIZONTAL_ND_BOUNDS = [i / args.horizontal_scale for i in HORIZONTAL_ND_BOUNDS]
nd_class = defaultdict(list)

# Function to determine the bin for each element
def find_bin_for_values(values, bounds):
    bins = []
    for value in values:
        bin_index = 0
        for bound in bounds:
            if value <= bound:
                break
            bin_index += 1
        bins.append(bin_index)
    return bins

def find_bins_for_row(data, vertical_bounds, horizontal_bounds, rule_name):
    # Convert string representation of list to actual list
    nd = ast.literal_eval(data)
    vertical_nd = nd[:8]
    horizontal_nd = nd[8:]
    if 'class' not in rule_name:
        vertical = find_bin_for_values(vertical_nd, vertical_bounds)
        horizontal = find_bin_for_values(horizontal_nd, horizontal_bounds)
        return vertical, horizontal
    else:
        return vertical_nd, horizontal_nd

# Create DataFrame
df = pd.read_csv(args.file_name)
# Extracting densities from each row
for i in range(len(df[f'{args.rule_name}.target_rule'])):
    row = df[f'{args.rule_name}.orig_rule'][i]
    orig_vertical_bins, orig_horizontal_bins = find_bins_for_row(row, VERTICAL_ND_BOUNDS, HORIZONTAL_ND_BOUNDS, args.rule_name)
    nd_class['orig_nd_vertical'].append(orig_vertical_bins)
    nd_class['orig_nd_horizontal'].append(orig_horizontal_bins)     
    row = df[f'{args.rule_name}.target_rule'][i]
    target_vertical_bins, target_horizontal_bins = find_bins_for_row(row, VERTICAL_ND_BOUNDS, HORIZONTAL_ND_BOUNDS, args.rule_name)
    nd_class['target_nd_vertical'].append(target_vertical_bins)
    nd_class['target_nd_horizontal'].append(target_horizontal_bins)   
    row = df[f'{args.rule_name}.gen_rule'][i]
    gen_vertical_bins, gen_horizontal_bins = find_bins_for_row(row, VERTICAL_ND_BOUNDS, HORIZONTAL_ND_BOUNDS, args.rule_name)
    nd_class['gen_nd_vertical'].append(gen_vertical_bins)
    nd_class['gen_nd_horizontal'].append(gen_horizontal_bins)
    vertical_loss = (np.array(target_vertical_bins) != np.array(gen_vertical_bins)).mean()
    horizontal_loss = (np.array(target_horizontal_bins) != np.array(gen_horizontal_bins)).mean()
    nd_class['vertical_nd.loss'].append(vertical_loss)
    nd_class['horizontal_nd.loss'].append(horizontal_loss)

error = pd.DataFrame(nd_class)
error.to_csv(args.file_name.replace('results', 'error'), index=False)

vt_error = error['vertical_nd.loss'].mean()
hr_error = error['horizontal_nd.loss'].mean()
mean_error = (error['vertical_nd.loss'].mean() + error['horizontal_nd.loss'].mean()) / 2

print(f"vertical_nd error: {vt_error}, horizontal_nd error: {hr_error}, mean error: {mean_error}")
