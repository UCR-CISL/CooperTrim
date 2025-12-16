import cv2
import os
import numpy as np

def merge_images_to_video(folder1, folder2,  output_video_path, fps=30, folder3=None):
    """
    Merges images from three folders into a grid (3 rows) and creates a video.

    Args:
        folder1 (str): Path to the first folder of images.
        folder2 (str): Path to the second folder of images.
        folder3 (str): Path to the third folder of images.
        output_video_path (str): Path to save the output video.
        fps (int): Frames per second for the video.
    """
    # Get sorted lists of image filenames from each folder
    images1 = sorted(os.listdir(folder1))
    images2 = sorted(os.listdir(folder2))
    # images3 = sorted(os.listdir(folder3))

    # Ensure all folders have the same number of images
    # assert len(images1) == len(images2) == len(images3), "Folders must have the same number of images"
    assert len(images1) == len(images2), "Folders must have the same number of images"

    # Load the first image to get dimensions
    sample_img1 = cv2.imread(os.path.join(folder1, images1[0]))
    height, width, _ = sample_img1.shape

    # Define video writer
    video_height = height * 3  # 3 rows
    video_width = width       # 1 column
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (video_width, video_height))

    # Process each set of images
    x=1
    # for img_name1, img_name2, img_name3 in zip(images1, images2, images3):
    for img_name1, img_name2 in zip(images1, images2):
        print(f"Processing frame {x} of {len(images1)}")
        x+=1
        # Read images
        img1 = cv2.imread(os.path.join(folder1, img_name1))
        img2 = cv2.imread(os.path.join(folder2, img_name2))
        # img3 = cv2.imread(os.path.join(folder3, img_name3))

        # Ensure all images have the same dimensions
        img1 = cv2.resize(img1, (width, height))
        img2 = cv2.resize(img2, (width, height))
        # img3 = cv2.resize(img3, (width, height))

        # Stack images vertically (3 rows)
        # merged_frame = np.vstack([img1, img2, img3])
        merged_frame = np.vstack([img1, img2])

        # Write the frame to the video
        out.write(merged_frame)

    # Release the video writer
    out.release()
    print(f"Video saved at: {output_video_path}")

if __name__ == "__main__":
    # Define paths to the folders
    folder1 = "/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/eval_merged_images/cobevt_50"  # Replace with the actual path to folder1
    folder2 = "/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/eval_merged_images/cobevt_50"  # Replace with the actual path to folder2
    folder3 = None  # Replace with the actual path to folder3

    # Define the output video path
    output_video_path = "/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/eval_merged_images/cobevt_video/video.mp4"  # Replace with the desired output path

    # Frames per second for the video
    fps = 30

    # Merge images and create the video
    merge_images_to_video(folder1, folder2, output_video_path, fps, folder3)
