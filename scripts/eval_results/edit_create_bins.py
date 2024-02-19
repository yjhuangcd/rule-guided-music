import pandas as pd
import numpy as np
import ast
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--file_name', type=str, default='loggings/edit_table/nd_500_cls_2/results.csv',
                    help='Path to the folder that contains generated samples for rule guidance')
parser.add_argument('--bins', type=int, default=8,
                    help='Str used to split out basenames')
args = parser.parse_args()

# Create DataFrame
df = pd.read_csv(args.file_name)

# Re-processing the note density data from the DataFrame to include the bounds
def process_note_density_data_with_bounds(df):
    # Initialize lists for vertical and horizontal densities
    vertical_densities = []
    horizontal_densities = []

    # Extracting densities from each row
    for row in df['note_density.target_rule']:
        # Convert string representation of list to actual list
        densities = ast.literal_eval(row)
        vertical_densities.extend(densities[:8])
        horizontal_densities.extend(densities[8:])

    # Function to create bins, count samples, and find bounds
    def create_bins_and_find_bounds(data):
        data_sorted = sorted(data)
        bins = np.array_split(data_sorted, 8)
        bins_count = [len(bin) for bin in bins]
        bins_bounds = [(bin[0], bin[-1]) for bin in bins if len(bin) > 0]
        return bins, bins_count, bins_bounds

    # Creating bins and finding bounds
    vertical_bins, vertical_bins_count, vertical_bins_bounds = create_bins_and_find_bounds(vertical_densities)
    horizontal_bins, horizontal_bins_count, horizontal_bins_bounds = create_bins_and_find_bounds(horizontal_densities)

    return vertical_bins_count, vertical_bins_bounds, horizontal_bins_count, horizontal_bins_bounds

# Process the data and get the bounds
vertical_bins_count, vertical_bins_bounds, horizontal_bins_count, horizontal_bins_bounds = process_note_density_data_with_bounds(df)

# Display the bins, their sample counts, and bounds
print("Vertical Note Density Bins:")
for i, (count, bounds) in enumerate(zip(vertical_bins_count, vertical_bins_bounds), 1):
    print(f"Bin {i} (Samples: {count}, Bounds: {bounds})")

print("\nHorizontal Note Density Bins:")
for i, (count, bounds) in enumerate(zip(horizontal_bins_count, horizontal_bins_bounds), 1):
    print(f"Bin {i} (Samples: {count}, Bounds: {bounds})")
