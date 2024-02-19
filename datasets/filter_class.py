import pandas as pd
import argparse

def filter_and_save_csv(file_path, class_label):
    """
    Filters a CSV file to keep only the rows where the 'classes' column equals the specified class label.
    Saves the filtered DataFrame to a new CSV file.

    :param file_path: Path to the original CSV file.
    :param class_label: The class label to filter by.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Filter out rows where 'classes' equals the specified class_label
    filtered_df = df[df['classes'] == class_label]

    # Save the filtered DataFrame to a new CSV file
    # The new file name is the original file name with '_cls_<class_label>' appended before the file extension
    new_file_path = file_path.replace('.csv', f'_cls_{class_label}.csv')
    filtered_df.to_csv(new_file_path, index=False)

    print(f"Filtered CSV saved as: {new_file_path}")

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Filter a CSV file by class and save to a new file.")
    parser.add_argument("--file_path", type=str, help="Path to the original CSV file")
    parser.add_argument("--class_label", type=int, help="The class label to filter by")

    # Parse arguments
    args = parser.parse_args()

    # Call the function with the provided arguments
    filter_and_save_csv(args.file_path, args.class_label)

if __name__ == "__main__":
    main()
