import numpy as np

# File path
file_path = '/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/coopertrim_cobevt_dyn_loss10_channel_all.txt'
# file_path = '/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/channel_usage_cobevt_CA_dyn.txt'

# Initialize a list to store percentages
percentages = []

# Read the data from the file
with open(file_path, 'r') as file:
    lines = file.readlines()
    # Start from the second line to ignore the first line
    for line in lines[1:]:
        # Extract the percentage from each line
        _, percentage = line.strip('()\n').split(',')
        percentages.append(float(percentage))  # Convert percentage to float


# Calculate statistics
mean_percentage = np.mean(percentages)
std_dev_percentage = np.std(percentages)
min_percentage = np.min(percentages)
max_percentage = np.max(percentages)

# Print the results
print(f"Mean of Request Percentage: {mean_percentage:.2f}")
print(f"Standard Deviation of Request Percentage: {std_dev_percentage:.2f}")
print(f"Minimum Request Percentage: {min_percentage:.2f}")
print(f"Maximum Request Percentage: {max_percentage:.2f}")
