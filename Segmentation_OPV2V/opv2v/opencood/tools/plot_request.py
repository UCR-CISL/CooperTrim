# import matplotlib.pyplot as plt
# import numpy as np

# # File paths for the two datasets
# # file_path_1 = '/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/channel_usage_attfuse_CA_dyn.txt'
# file_path_1 = '/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/channel_usage_cobevt_dyn.txt'
# # file_path_3 = '/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/channel_usage_fcooper_CA_dyn.txt'
# # file_path_4 = '/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/channel_usage_disconet_CA_dyn.txt'


# # Function to read and process data from a file
# def read_and_process_data(file_path, window_size=100):
#     frame_ids = []
#     percentages = []
    
#     # Read the data from the file
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#         for line in lines:
#             # Extract frame ID and percentage from each line
#             frame_id, percentage = line.strip('()\n').split(',')
#             frame_ids.append(int(frame_id))  # Convert frame ID to integer
#             percentages.append(float(percentage))  # Convert percentage to float
    
#     # Apply a running average (smoothing) with the specified window size
#     percentages = np.convolve(percentages, np.ones(window_size) / window_size, mode='valid')
    
#     # Adjust frame IDs to match the smoothed data length
#     frame_ids = frame_ids[:len(percentages)]
    
#     return frame_ids, percentages

# # Read and process data from both files
# frame_ids_1, percentages_1 = read_and_process_data(file_path_1)
# # frame_ids_2, percentages_2 = read_and_process_data(file_path_2)
# # frame_ids_3, percentages_3 = read_and_process_data(file_path_3)
# # frame_ids_4, percentages_4 = read_and_process_data(file_path_4)

# # Plot the data from both files
# plt.figure(figsize=(10, 6))

# # Plot data 
# plt.plot(frame_ids_1, percentages_1, linestyle='-', color='r', label='CoBEVT')
# # plt.plot(frame_ids_2, percentages_1, linestyle='-', color='b', label='AttFuse')
# # plt.plot(frame_ids_3, percentages_3, linestyle='-', color='g', label='Fcooper')
# # plt.plot(frame_ids_4, percentages_4, linestyle='-', color='pink', label='Disconet')

# # Add labels, title, and legend
# plt.xlabel('Frame ID', fontsize=12)
# plt.ylabel('Request Percentage (%)', fontsize=12)
# plt.title('Frame-wise Request Percentage (Dynamic)', fontsize=14)
# plt.legend()
# plt.grid(True)

# # Save the plot as an image
# output_image_path = '/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/eval_results/frame_request_percentage_dyn.png'
# plt.savefig(output_image_path, dpi=300, bbox_inches='tight')  # Save with high resolution

# # Show the plot
# plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Function to read file data
def read_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # Remove parentheses and newline, then split by comma
            line = line.strip().replace('(', '').replace(')', '')
            frame, value = map(float, line.split(','))
            data.append((int(frame), value))
    return data

def read_file_static(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # Remove parentheses and newline, then split by comma
            line = line.strip().replace('(', '').replace(')', '')
            frame, value1, value2 = map(float, line.split(','))
            data.append((int(frame), value1, value2))
    return data


# Read data from the files
# # file1_static = read_file('/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/channel_usage_cobevt_st_base.txt')  # Replace with the actual path to static network usage
file2_static = read_file_static('/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/fcooper_st_iou_base.txt')  # Replace with the actual path to static IOU data

file1_l6 = read_file('/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/channel_usage_fcooper_st_lagrange.txt')  # Replace with the actual path to dynamic network usage
file2_l6 = read_file_static('/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/fcooper_st_iou_lagrange.txt')  # Replace with the actual path to dynamic IOU data

# Adjust frames in File 2 by adding 1
adjusted_static_data = [(frame + 1, riou, liou) for frame, riou, liou in file2_static]  # Static data adjustment
adjusted_l6_data = [(frame + 1, riou, liou) for frame, riou, liou in file2_l6]  # Dynamic data adjustment

# Extract data for plotting
frames_static, road_iou_static, lane_iou_static = zip(*adjusted_static_data)  # Static data extraction
frames_l6, road_iou_l6, lane_iou_l6 = zip(*adjusted_l6_data)  # Dynamic data extraction

# network_usage_static = [usage for _, usage in file1_static]  # Network usage for static frames
network_usage_l6 = [usage for _, usage in file1_l6]  # Network usage for dynamic frames


def gaussian_smoothing(data, sigma):
    kernel_size = int(6 * sigma + 1)  # Kernel size based on sigma
    kernel = np.exp(-np.linspace(-3*sigma, 3*sigma, kernel_size)**2 / (2*sigma**2))
    kernel /= sum(kernel)  # Normalize the kernel
    smoothed_data = np.convolve(data, kernel, mode='same')
    return smoothed_data

# Apply Gaussian smoothing to dynamic IOU data
sigma=10
road_iou_l6 = gaussian_smoothing(road_iou_l6, sigma=sigma)  # Smoothen Road IOU (Dynamic)
lane_iou_l6 = gaussian_smoothing(lane_iou_l6, sigma=sigma) 
network_usage_l6 = gaussian_smoothing(network_usage_l6, sigma=sigma) 
road_iou_static = gaussian_smoothing(road_iou_static, sigma=sigma)  # Smoothen Road IOU (Dynamic)
lane_iou_static = gaussian_smoothing(lane_iou_static, sigma=sigma) 
# network_usage_static = gaussian_smoothing(network_usage_static, sigma=sigma) 


# Plot Lane IOUs (Static and Dynamic)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(frames_static, lane_iou_static, network_usage_static, c='b', marker='o', label='Base Lane IOU')  # Static Lane IOU
# ax.scatter(frames_l6, lane_iou_l6, network_usage_l6, c='g', marker='^', label='L5 Lane IOU')  # Dynamic Lane IOU
# ax.set_xlabel('Frame')  # X-axis label
# ax.set_ylabel('Lane IOU')  # Y-axis label
# ax.set_zlabel('Network Usage')  # Z-axis label
# ax.legend()  # Add legend
# plt.savefig('/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/eval_results/3d_plot_lane_ious_combined_L5.png', dpi=300)  # Save as a high-resolution image
# plt.close()  # Close the plot to avoid displaying it

# # Plot Road IOUs (Static and Dynamic)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(frames_static, road_iou_static, network_usage_static, c='r', marker='o', label='Base Road IOU')  # Static Road IOU
# ax.scatter(frames_l6, road_iou_l6, network_usage_l6, c='y', marker='^', label='L5 Road IOU')  # Dynamic Road IOU
# ax.set_xlabel('Frame')  # X-axis label
# ax.set_ylabel('Road IOU')  # Y-axis label
# ax.set_zlabel('Network Usage')  # Z-axis label
# ax.legend()  # Add legend
# plt.savefig('/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/eval_results/3d_plot_road_ious_combined_L5.png', dpi=300)  # Save as a high-resolution image
# plt.close()  # Close the plot to avoid displaying it


# # Plot 2D Road IOU vs Frames
# plt.figure(figsize=(10, 6))
# plt.plot(frames_static, road_iou_static, linestyle='-', color='r', label='Base Road IOU')  # Static Road IOU
# plt.plot(frames_l6, road_iou_l6, linestyle='--', color='y', label='L6 Road IOU')  # Dynamic Road IOU
# plt.xlabel('Frame', fontsize=12)
# plt.ylabel('Road IOU', fontsize=12)
# plt.title('Road IOU vs Frame', fontsize=14)
# plt.legend()
# plt.grid(True)
# plt.savefig('/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/eval_results/2d_plot_road_ious_cobevt.png', dpi=300, bbox_inches='tight')  # Save the plot
# plt.close()  # Close the plot to avoid displaying it

# # Plot 2D Lane IOU vs Frames
# plt.figure(figsize=(10, 6))
# plt.plot(frames_static, lane_iou_static, linestyle='-', color='b', label='Base Lane IOU')  # Static Lane IOU
# plt.plot(frames_l6, lane_iou_l6, linestyle='--', color='g', label='L6 Lane IOU')  # Dynamic Lane IOU
# plt.xlabel('Frame', fontsize=12)
# plt.ylabel('Lane IOU', fontsize=12)
# plt.title('Lane IOU vs Frame', fontsize=14)
# plt.legend()
# plt.grid(True)
# plt.savefig('/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/eval_results/2d_plot_lane_ious_cobevt.png', dpi=300, bbox_inches='tight')  # Save the plot
# plt.close()  # Close the plot to avoid displaying it

# Calculate the difference between network usage data
# network_usage_diff = [static - dynamic for static, dynamic in zip(network_usage_static, network_usage_l6)]

# Plot 2D Road IOU vs Frames
plt.figure(figsize=(10, 6))
plt.plot(frames_static, road_iou_static, linestyle='-', color='r', label='Base Road IOU')  # Static Road IOU
plt.plot(frames_l6, road_iou_l6, linestyle='-', color='y', label='Lagrange Road IOU')  # Dynamic Road IOU
plt.plot(frames_static, network_usage_l6, linestyle='--', color='purple', label='Network Usage Lagrange')  # Network Usage Difference
plt.xlabel('Frame', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.title('Road IOU and Network Usage vs Frame', fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig('/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/eval_results/2d_plot_road_ious_fcooper_st.png', dpi=300, bbox_inches='tight')  # Save the plot
plt.close()  # Close the plot to avoid displaying it

# Plot 2D Lane IOU vs Frames
plt.figure(figsize=(10, 6))
plt.plot(frames_static, lane_iou_static, linestyle='-', color='r', label='Base Lane IOU')  # Static Lane IOU
plt.plot(frames_l6, lane_iou_l6, linestyle='-', color='y', label='Lagrange Lane IOU')  # Dynamic Lane IOU
plt.plot(frames_static, network_usage_l6, linestyle='--', color='purple', label='Network Usage Lagrange')  # Network Usage Difference
plt.xlabel('Frame', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.title('Lane IOU and Network Usage vs Frame', fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig('/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/eval_results/2d_plot_lane_ious_fcooper_st.png', dpi=300, bbox_inches='tight')  # Save the plot
plt.close()  # Close the plot to avoid displaying it

###############DYNAMIC PLOTTING#####################

# # # Read data from the files

file2_dynamic = read_file('/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/fcooper_dyn_iou_base.txt')  # Replace with the actual path to static IOU data
file1_l6 = read_file('/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/channel_usage_fcooper_dyn_lagrange.txt')  # Replace with the actual path to dynamic network usage
file2_l6 = read_file('/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/fcooper_dyn_iou_lagrange.txt')  # Replace with the actual path to dynamic IOU data

# Adjust frames in File 2 by adding 1
adjusted_dynamic_data = [(frame + 1, riou) for frame, riou in file2_dynamic]  # Static data adjustment
adjusted_l6_data = [(frame + 1, riou) for frame, riou in file2_l6]  # Dynamic data adjustment

# Extract data for plotting
frames_dynamic, iou_dynamic = zip(*adjusted_dynamic_data)  # Static data extraction
frames_l6, iou_l6 = zip(*adjusted_l6_data)  # Dynamic data extraction

network_usage_l6 = [usage for _, usage in file1_l6]  # Network usage for dynamic frames


def gaussian_smoothing(data, sigma):
    kernel_size = int(6 * sigma + 1)  # Kernel size based on sigma
    kernel = np.exp(-np.linspace(-3*sigma, 3*sigma, kernel_size)**2 / (2*sigma**2))
    kernel /= sum(kernel)  # Normalize the kernel
    smoothed_data = np.convolve(data, kernel, mode='same')
    return smoothed_data

# Apply Gaussian smoothing to dynamic IOU data
sigma=10
iou_l6 = gaussian_smoothing(iou_l6, sigma=sigma)  # Smoothen Road IOU (Dynamic)
network_usage_l6 = gaussian_smoothing(network_usage_l6, sigma=sigma) 
iou_dynamic = gaussian_smoothing(iou_dynamic, sigma=sigma)  # Smoothen Road IOU (Dynamic)




# Plot Lane IOUs (Static and Dynamic)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(frames_static, lane_iou_static, network_usage_static, c='b', marker='o', label='Base Lane IOU')  # Static Lane IOU
# ax.scatter(frames_l6, lane_iou_l6, network_usage_l6, c='g', marker='^', label='L5 Lane IOU')  # Dynamic Lane IOU
# ax.set_xlabel('Frame')  # X-axis label
# ax.set_ylabel('Lane IOU')  # Y-axis label
# ax.set_zlabel('Network Usage')  # Z-axis label
# ax.legend()  # Add legend
# plt.savefig('/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/eval_results/3d_plot_lane_ious_combined_L5.png', dpi=300)  # Save as a high-resolution image
# plt.close()  # Close the plot to avoid displaying it

# # Plot Road IOUs (Static and Dynamic)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(frames_static, road_iou_static, network_usage_static, c='r', marker='o', label='Base Road IOU')  # Static Road IOU
# ax.scatter(frames_l6, road_iou_l6, network_usage_l6, c='y', marker='^', label='L5 Road IOU')  # Dynamic Road IOU
# ax.set_xlabel('Frame')  # X-axis label
# ax.set_ylabel('Road IOU')  # Y-axis label
# ax.set_zlabel('Network Usage')  # Z-axis label
# ax.legend()  # Add legend
# plt.savefig('/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/eval_results/3d_plot_road_ious_combined_L5.png', dpi=300)  # Save as a high-resolution image
# plt.close()  # Close the plot to avoid displaying it


# # Plot 2D Road IOU vs Frames
# plt.figure(figsize=(10, 6))
# plt.plot(frames_static, road_iou_static, linestyle='-', color='r', label='Base Road IOU')  # Static Road IOU
# plt.plot(frames_l6, road_iou_l6, linestyle='--', color='y', label='L6 Road IOU')  # Dynamic Road IOU
# plt.xlabel('Frame', fontsize=12)
# plt.ylabel('Road IOU', fontsize=12)
# plt.title('Road IOU vs Frame', fontsize=14)
# plt.legend()
# plt.grid(True)
# plt.savefig('/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/eval_results/2d_plot_road_ious_cobevt.png', dpi=300, bbox_inches='tight')  # Save the plot
# plt.close()  # Close the plot to avoid displaying it

# # Plot 2D Lane IOU vs Frames
# plt.figure(figsize=(10, 6))
# plt.plot(frames_static, lane_iou_static, linestyle='-', color='b', label='Base Lane IOU')  # Static Lane IOU
# plt.plot(frames_l6, lane_iou_l6, linestyle='--', color='g', label='L6 Lane IOU')  # Dynamic Lane IOU
# plt.xlabel('Frame', fontsize=12)
# plt.ylabel('Lane IOU', fontsize=12)
# plt.title('Lane IOU vs Frame', fontsize=14)
# plt.legend()
# plt.grid(True)
# plt.savefig('/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/eval_results/2d_plot_lane_ious_cobevt.png', dpi=300, bbox_inches='tight')  # Save the plot
# plt.close()  # Close the plot to avoid displaying it

# Calculate the difference between network usage data
# network_usage_diff = [static - dynamic for static, dynamic in zip(network_usage_static, network_usage_l6)]

# Plot 2D Road IOU vs Frames
plt.figure(figsize=(10, 6))
plt.plot(frames_dynamic, iou_dynamic, linestyle='-', color='r', label='Base Dynamic IOU')  # Static Road IOU
plt.plot(frames_l6, iou_l6, linestyle='-', color='y', label='Lagrange Dynamic IOU')  # Dynamic Road IOU
plt.plot(frames_dynamic, network_usage_l6, linestyle='--', color='purple', label='Network Usage Lagrange')  # Network Usage Difference
plt.xlabel('Frame', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.title('Dynamic IOU and Network Usage vs Frame', fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig('/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/eval_results/2d_plot_fcooper_dyn.png', dpi=300, bbox_inches='tight')  # Save the plot
plt.close()  # Close the plot to avoid displaying it

print("Plots generated successfully.")

