import pandas as pd
import numpy as np
import torch

# Step 1: Read the .txt file into a DataFrame
file_path = "/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/attfuse_st_compression_stats_8x_lossy.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)

# Step 2: Calculate the average of 'Avg Original Size per CAV' and 'Avg Compressed Size per CAV'
avg_original_per_cav = df['Avg Original Size per CAV (bytes)'].mean()
avg_compressed_per_cav = df['Avg Compressed Size per CAV (bytes)'].mean()

print(f"Average Original Size per CAV: {avg_original_per_cav:.2f} bytes")
print(f"Average Compressed Size per CAV: {avg_compressed_per_cav:.2f} bytes")

# Step 3: Calculate the percentage of compressed size with respect to original size
compression_percentage = (avg_compressed_per_cav / avg_original_per_cav) * 100
print(f"Percentage of Compressed Size relative to Original Size: {compression_percentage:.2f}%")

# Step 4: Calculate initial size for a tensor of size [128, 32, 32]
# Assuming the tensor is of type float32 (common for PyTorch tensors, 4 bytes per element)
tensor = torch.randn(128, 32, 32)  # Dummy tensor to get size calculation
original_size = tensor.element_size() * tensor.nelement()
print(f"Initial Size for tensor of shape [128, 32, 32]: {original_size} bytes")
