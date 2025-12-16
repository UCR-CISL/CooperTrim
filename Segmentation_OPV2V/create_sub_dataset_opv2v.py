import os
import shutil
import random

def copy_dataset(src_dir, dest_dir, train_percentage=0.4):
    """
    Copies 40% of train data folders and 100% of test and validate data folders to a new location.

    Parameters:
    - src_dir (str): Path to the source dataset directory.
    - dest_dir (str): Path to the destination dataset directory.
    - train_percentage (float): Percentage of train data folders to copy (default is 40%).
    """
    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Iterate over dataset subdirectories
    for folder in ['train', 'test', 'validate']:
        src_folder_path = os.path.join(src_dir, folder)
        dest_folder_path = os.path.join(dest_dir, folder)

        if not os.path.exists(src_folder_path):
            print(f"Source folder '{src_folder_path}' does not exist. Skipping.")
            continue

        os.makedirs(dest_folder_path, exist_ok=True)

        # Get list of all subfolders in the current dataset folder
        subfolders = [f for f in os.listdir(src_folder_path) if os.path.isdir(os.path.join(src_folder_path, f))]

        if folder == 'train':
            # Randomly select 40% of train subfolders
            num_to_copy = int(len(subfolders) * train_percentage)
            subfolders_to_copy = random.sample(subfolders, num_to_copy)
        else:
            # Copy all subfolders for test and validate
            subfolders_to_copy = subfolders

        # Copy selected subfolders to destination
        print(f"Copying {len(subfolders_to_copy)} subfolders from '{src_folder_path}' to '{dest_folder_path}'...")
        for subfolder in subfolders_to_copy:
            src_subfolder_path = os.path.join(src_folder_path, subfolder)
            dest_subfolder_path = os.path.join(dest_folder_path, subfolder)
            shutil.copytree(src_subfolder_path, dest_subfolder_path)
            print(f"Copied '{src_subfolder_path}' to '{dest_subfolder_path}'.")

if __name__ == "__main__":
    # Example usage
    src_dataset_dir = "/data/HangQiu/data/OPV2V"
    dest_dataset_dir = "/data/HangQiu/proj/autonet-RL/sub_OPV2V"

    copy_dataset(src_dataset_dir, dest_dataset_dir)
