"""
Implementation of Brady Zhou's cross view transformer
"""
import einops
import numpy as np
import torch.nn as nn
import torch
from einops import rearrange
from opencood.models.sub_modules.fax_modules import FAXModule
from opencood.models.backbones.resnet_ms import ResnetEncoder
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.sub_modules.bev_seg_head import BevSegHead
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fusion_modules.swap_fusion_modules import \
    SwapFusionEncoder
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    get_transformation_matrix, warp_affine, get_roi_and_cav_mask, \
    get_discretized_transformation_matrix

#shilpa bev dim match
# import torch.nn.functional as F
\
#shilpa rebuttal localization
import random

#shilpa entropy
from scipy.stats import entropy

#shilpa compression
import zlib
import csv
import os




class STTF(nn.Module):
    def __init__(self, args):
        super(STTF, self).__init__()
        self.discrete_ratio = args['resolution']
        self.downsample_rate = args['downsample_rate']

    def forward(self, x, spatial_correction_matrix):
        """
        Transform the bev features to ego space.

        Parameters
        ----------
        x : torch.Tensor
            B L C H W
        spatial_correction_matrix : torch.Tensor
            Transformation matrix to ego

        Returns
        -------
        The bev feature same shape as x but with transformation
        """
        dist_correction_matrix = get_discretized_transformation_matrix(
            spatial_correction_matrix, self.discrete_ratio,
            self.downsample_rate)

        # transpose and flip to make the transformation correct
        x = rearrange(x, 'b l c h w  -> b l c w h')
        x = torch.flip(x, dims=(4,))
        # Only compensate non-ego vehicles
        B, L, C, H, W = x.shape

        T = get_transformation_matrix(
            dist_correction_matrix[:, :, :, :].reshape(-1, 2, 3), (H, W))
        cav_features = warp_affine(x[:, :, :, :, :].reshape(-1, C, H, W), T,
                                   (H, W))
        cav_features = cav_features.reshape(B, -1, C, H, W)

        # flip and transpose back
        x = cav_features
        x = torch.flip(x, dims=(4,))
        x = rearrange(x, 'b l c w h -> b l h w c')

        return x
     


class CorpBEVT(nn.Module):
    def __init__(self, config):
        super(CorpBEVT, self).__init__()
        #shilpa max_cav change inference
        self.max_cav = config['max_cav']
        # encoder params
        self.encoder = ResnetEncoder(config['encoder'])

        # cvm params
        fax_params = config['fax']
        fax_params['backbone_output_shape'] = self.encoder.output_shapes
        self.fax = FAXModule(fax_params)

        if config['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(128, config['compression'])
        else:
            self.compression = False

        # spatial feature transform module
        self.downsample_rate = config['sttf']['downsample_rate']
        self.discrete_ratio = config['sttf']['resolution']
        self.use_roi_mask = config['sttf']['use_roi_mask']
        self.sttf = STTF(config['sttf'])
        #shilpa
        # self.find_transformed_indices = STTF(config['sttf']).find_transformed_indices

        # spatial fusion
        self.fusion_net = SwapFusionEncoder(config['fax_fusion'])

        # decoder params
        decoder_params = config['decoder']
        # decoder for dynamic and static differet
        self.decoder = NaiveDecoder(decoder_params)

        self.target = config['target']
        self.seg_head = BevSegHead(self.target,
                                   config['seg_head_dim'],
                                   config['output_class'])
        
        #shilpa entropy
        # self.prev_avg_entropy = None
        #shilpa prev feature for uncertainty improvement
        self.prev_fused_feature = None

        #shilpa grad cam
        self.frame_counter = 0

        #shilpa rebuttal latency
        # Initialize variables for simulating data delay/latency
        self.ids = []  # To store selected IDs for delayed data
        self.previous_perception_data = None  # To store previous data for selected IDs
        self.counter = 0  # To track frames since data was stored
        self.delay_frames = 0  # Number of frames to wait before replacement

    #self compression
    # Function to quantize tensor for lossy compression (to approximate desired compression factor)
    # def quantize_tensor(self, tensor, factor):
    #     """
    #     Reduce precision of tensor values to approximate a target compression factor.
    #     Higher factor means more aggressive quantization (more lossy).
    #     """
    #     # Scale factor to control quantization (higher factor -> lower precision)
    #     quantization_levels = int(256 / factor)  # Rough approximation
    #     if quantization_levels < 1:
    #         quantization_levels = 1
    #     # Quantize by rounding to fewer levels (works with PyTorch tensor)
    #     quantized = torch.round(tensor * quantization_levels) / quantization_levels
    #     return quantized

    def quantize_tensor(self, tensor, factor):
        """
        Reduce precision of tensor values for lossy compression with a more aggressive approach.
        Higher factor means more aggressive quantization (more lossy).
        """
        # Check if tensor is empty
        if tensor.numel() == 0:
            return tensor  # Return unchanged if empty
        
        # Determine the range of tensor values to map quantization effectively
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        value_range = tensor_max - tensor_min
        
        if value_range == 0:  # Avoid division by zero if all values are the same
            return tensor
        
        # Calculate quantization levels based on factor (more aggressive reduction)
        quantization_levels = max(2, int(256 / (factor ** 0.5)))  # Adjusted to be more aggressive
        if quantization_levels < 2:
            quantization_levels = 2
        
        # Normalize tensor to [0, quantization_levels], round, and scale back to original range
        normalized = (tensor - tensor_min) / value_range * (quantization_levels - 1)
        quantized = torch.round(normalized)
        dequantized = quantized / (quantization_levels - 1) * value_range + tensor_min
        
        return dequantized


    # Function to compress the tensor (lossless or lossy based on mode)
    def compress_tensor(self, tensor, compression_factor=None, mode='lossless'):
        """
        Compress tensor with zlib. If mode='lossy', apply quantization based on compression_factor.
        compression_factor: Target factor like 2, 4, 8, 10, 12 (used for quantization in lossy mode or level mapping in lossless)
        mode: 'lossless' (zlib only) or 'lossy' (quantization + zlib)
        """
        if mode == 'lossy' and compression_factor is not None:
            # Apply quantization to reduce data precision before compression
            tensor = self.quantize_tensor(tensor, compression_factor)
        
        # Convert PyTorch tensor to NumPy array for compression
        tensor_np = tensor.cpu().numpy()  # Ensure tensor is on CPU before conversion
        
        # Convert NumPy array to bytes
        tensor_bytes = tensor_np.tobytes()
        
        # Map compression factor to zlib level (0-9) for lossless mode if provided
        if compression_factor is not None and mode == 'lossless':
            # Rough mapping: higher factor -> higher compression level (though not exact)
            # level = min(9, max(0, int(compression_factor / 2)))  # Example mapping
            level = min(9, max(0, int(compression_factor))) 
        else:
            level = 9  # Default to maximum compression
        
        # Compress using zlib (lossless compression)
        compressed_data = zlib.compress(tensor_bytes, level=level)
        return compressed_data, tensor  # Return modified tensor if quantized

    # Function to decompress the data back to a tensor
    def decompress_tensor(self, compressed_data, original_shape, dtype=torch.float32):
        # Decompress the data
        decompressed_bytes = zlib.decompress(compressed_data)
        # Get the corresponding NumPy dtype from PyTorch dtype
        np_dtype = dtype.numpy_dtype if hasattr(dtype, 'numpy_dtype') else np.float32
        # Convert back to NumPy array with the original shape and dtype
        tensor_np = np.frombuffer(decompressed_bytes, dtype=np_dtype).reshape(original_shape)
        # Convert NumPy array back to PyTorch tensor
        return torch.from_numpy(tensor_np).to(dtype=dtype)

    # Function to calculate sizes and log to CSV with averages for multiple CAVs
    def log_compression_stats(self, original_tensor, compressed_data, compression_factor=None, mode='lossless', csv_filename="compression_stats.csv", num_cavs=1, is_lossless=True):
        """
        Calculate and log compression statistics, including averages per CAV if num_cavs > 1.
        num_cavs: Number of CAVs to calculate average data (default is 1, meaning no averaging).
        """
        # Calculate sizes in bytes for PyTorch tensor
        original_size = original_tensor.element_size() * original_tensor.nelement()
        compressed_size = len(compressed_data)
        gained_size = original_size - compressed_size
        actual_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        # Calculate averages if num_cavs > 1
        if num_cavs > 1:
            avg_original_size = original_size / num_cavs
            avg_compressed_size = compressed_size / num_cavs
            avg_gained_size = gained_size / num_cavs
            # Compression ratio remains the same as it's a ratio, not a size
        else:
            avg_original_size = original_size
            avg_compressed_size = compressed_size
            avg_gained_size = gained_size
        
        # # Print sizes for verification (both total and average if applicable)
        # print(f"\nResults for Compression Factor={compression_factor if compression_factor else 'Default'}, Mode={mode}:")
        # print(f"Total Original Size: {original_size} bytes")
        # print(f"Total Compressed Size: {compressed_size} bytes")
        # print(f"Total Gained Size (Space Saved): {gained_size} bytes")
        # print(f"Actual Compression Ratio: {actual_ratio:.2f}")
        # if num_cavs > 1:
        #     print(f"\nAverage per CAV (n={num_cavs}):")
        #     print(f"Average Original Size: {avg_original_size:.2f} bytes")
        #     print(f"Average Compressed Size: {avg_compressed_size:.2f} bytes")
        #     print(f"Average Gained Size (Space Saved): {avg_gained_size:.2f} bytes")
        # print("-" * 50)
        
        # Write to CSV (include both total and average if applicable)
        file_exists = os.path.isfile(csv_filename)
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Write header if file doesn't exist
            if not file_exists:
                writer.writerow(["Compression Factor", "Mode", "Num CAVs", "Total Original Size (bytes)", "Total Compressed Size (bytes)", "Total Gained Size (bytes)", "Actual Compression Ratio", "Avg Original Size per CAV (bytes)", "Avg Compressed Size per CAV (bytes)", "Avg Gained Size per CAV (bytes)", "Is_Lossless"])
            # Write data (if num_cavs == 1, averages are same as totals)
            writer.writerow([compression_factor if compression_factor else "Default", mode, num_cavs, original_size, compressed_size, gained_size, actual_ratio, avg_original_size, avg_compressed_size, avg_gained_size, int(is_lossless)])
        # print(f"Stats logged to {csv_filename}")

    def compress_features(self, selected_output_values_k,  num_cavs=1, factor=2, csv_filename="compression_stats.csv", mode="lossless"):
        """
        Compress features and log stats, with option to calculate averages for multiple CAVs.
        num_cavs: Number of CAVs to calculate average data (default is 1, meaning no averaging).
        """
        original_dtype = selected_output_values_k.dtype
        original_shape = selected_output_values_k.shape
        num_cavs = original_shape[1]
        
        # Test lossless compression with the specified factor
        # print("Testing Lossless Compression:")
        compressed_data, _ = self.compress_tensor(selected_output_values_k, compression_factor=factor, mode=mode)
        decompressed_tensor = self.decompress_tensor(compressed_data, original_shape, original_dtype)
        is_lossless = torch.allclose(selected_output_values_k, decompressed_tensor.to(device=selected_output_values_k.device), atol=1e-8)
        # print(f"Lossless Compression Verified for factor {factor}: {is_lossless}")
        self.log_compression_stats(selected_output_values_k, compressed_data, compression_factor=factor, mode=mode,csv_filename=csv_filename, num_cavs=num_cavs, is_lossless=is_lossless)
        
        # Test lossy compression with quantization to approximate desired factors (uncomment if needed)
        """
        print("\nTesting Lossy Compression (with Quantization):")
        compressed_data, modified_tensor = self.compress_tensor(selected_output_values_k, compression_factor=factor, mode='lossy')
        decompressed_tensor = self.decompress_tensor(compressed_data, original_shape, original_dtype)
        is_lossless = torch.allclose(selected_output_values_k, decompressed_tensor, atol=1e-8)
        print(f"Lossless Compression Verified for factor {factor} (lossy mode): {is_lossless}")
        self.log_compression_stats(selected_output_values_k, compressed_data, compression_factor=factor, mode='lossy', num_cavs=num_cavs)
        """
        return decompressed_tensor.to(device=selected_output_values_k.device)
    
    #shilpa rebuttal loss simulation
    def simulate_data_loss(self, x, record_len):
        """
        Simulates data loss in a tensor x of shape (b, l, c, h, w) within the valid record_len entries.
        
        Args:
            x (torch.Tensor): Input tensor of shape (b, l, c, h, w)
            record_len (int or torch.Tensor): Number of valid entries along the 'l' dimension for each batch.
                If int, same for all batch elements; if tensor, shape (b,) with per-batch values.
        
        Returns:
            torch.Tensor: Tensor with simulated data loss (corrupted entries set to 0)
            float: Selected loss percentage
        """
        # Get the shape of the tensor
        b, l, c, h, w = x.shape
        
        # Step 1: Randomly select loss percentage from [0, 1, 2, 5, 10]
        # loss_percent = random.choice([0, 1, 2, 5, 10])
        loss_percent = random.randint(0,10)  # Uniform random between 0 and 10
        print(f"Selected Loss Percentage: {loss_percent}%")
        
        # If loss_percent is 0, return the tensor unchanged
        if loss_percent == 0:
            return x, loss_percent
        
        # Step 2: Handle record_len (convert to tensor if it's an int)
        if isinstance(record_len, int):
            record_len = torch.full((b,), record_len, device=x.device)
        elif isinstance(record_len, torch.Tensor):
            record_len = record_len.to(x.device)
        else:
            raise ValueError("record_len must be an int or torch.Tensor")
        
        # Create a copy of the input tensor to modify
        x_corrupted = x.clone()
        
        # Step 3: Corrupt data for each batch element
        for i in range(b):
            # Get the valid length for this batch element
            valid_len = min(record_len[i].item(), l)  # Ensure it doesn't exceed l
            
            # Only proceed if record_len > 1 (other CAVs exist beyond ego vehicle)
            if valid_len <= 1:
                continue
            
            # Total number of CAVs to consider for corruption (exclude index 0)
            num_cavs = valid_len - 1
            
            # Total number of data elements across all CAVs (indices 1 to valid_len-1)
            total_elements = num_cavs * c * h * w
            
            # Calculate number of elements to corrupt based on loss percentage
            num_corrupt_elements = int(total_elements * loss_percent / 100.0)
            if num_corrupt_elements == 0:
                continue  # No corruption needed if num_corrupt_elements is 0
            
            # Step 4: Randomly select elements to corrupt
            # First, determine distribution across CAVs, channels, height, and width
            # Generate random indices for all dimensions
            cav_indices = torch.randint(1, valid_len, (num_corrupt_elements,), device=x.device)  # CAVs from 1 to valid_len-1
            c_indices = torch.randint(0, c, (num_corrupt_elements,), device=x.device)
            h_indices = torch.randint(0, h, (num_corrupt_elements,), device=x.device)
            w_indices = torch.randint(0, w, (num_corrupt_elements,), device=x.device)
            
            # Set the selected elements to 0 (corrupt the data)
            x_corrupted[i, cav_indices, c_indices, h_indices, w_indices] = 0.0
        
        return x_corrupted, loss_percent

    #shilpa epsilon greedy
    # def forward(self, batch_dict, ppo_agent=None):
    def forward(self, batch_dict, epoch, ppo_agent=None):
        x = batch_dict['inputs']
        b, l, m, _, _, _ = x.shape

        # shape: (B, max_cav, 4, 4)
        transformation_matrix = batch_dict['transformation_matrix']
        record_len = batch_dict['record_len']  # record_len is the number of CAVs in the scene

        #shilpa rebuttal localization
        # # Step 1: Sample alpha from uniform distribution [0, 1]
        # alpha = torch.FloatTensor(1).uniform_(0, 1).item()
        # # Step 2: Check if alpha < 0.2 to proceed with noise injection
        # if alpha < 1.0:
        #     # print("alpha < 0.2, proceeding with noise injection...")
        #     num_selections = min(3, record_len)    
        #     # Randomly select unique indices from valid range [0, record_len-1]
        #     # Using random.sample to avoid duplicates
        #     selected_indices = random.sample(range(record_len), num_selections)
        #     # print("Selected Indices for Noise Injection:", selected_indices)

        #     # Step 4: For each selected index, generate random noise and add to Tx, Ty, Tz
        #     for idx in selected_indices:
        #         if idx<record_len:
        #             # Generate 3 random noise values between -20.0 and 20.0
        #             noise = torch.FloatTensor(3).uniform_(-1.0, 1.0)
        #             a, b, c = noise[0], noise[1], noise[2]
        #             # print(f"Index {idx}: Adding noise (a={a:.2f}, b={b:.2f}, c={c:.2f}) to Tx, Ty, Tz")
                    
        #             # Add noise to Tx (position [0,3]), Ty ([1,3]), Tz ([2,3]) at batch=0, seq=idx
        #             transformation_matrix[0, idx, 0, 3] += a
        #             transformation_matrix[0, idx, 1, 3] += b
        #             transformation_matrix[0, idx, 2, 3] += c
        # # else:
        # #     print("alpha >= 0.2, skipping noise injection.")

        x = self.encoder(x)
        batch_dict.update({'features': x})
      
        #shilpa select threshold
        # orig_bev_data_from_all_cav, selected_indices, select_threhold, percentage_selected = self.fax(batch_dict, self.prev_fused_feature)

        #shilpa epsilon greedy
        orig_bev_data_from_all_cav, selected_indices, select_threhold, percentage_selected = self.fax(batch_dict, epoch, self.prev_fused_feature)
        
        #shilpa rebuttal latency
        # # Simulate data delay/latency with 30% probability
        # beta = 1.0
        # if random.random() < beta and self.counter == 0:  # Only store new data if no delay is in progress
        #     # Get possible IDs excluding 0th ID
        #     possible_ids = list(range(1, record_len[0]))  # Assuming record_len is a tensor/list with batch size 1
        #     if possible_ids and record_len[0]>1:  # Ensure there are IDs to select
        #         # Randomly select a subset of IDs
        #         num_ids_to_select = random.randint(1, len(possible_ids)) if len(possible_ids) > 1 else 1
        #         self.ids = random.sample(possible_ids, num_ids_to_select)
        #         # Store corresponding data from orig_bev_data_from_all_cav
        #         # Assuming orig_bev_data_from_all_cav shape is [record_len, ch, h, w]
        #         self.previous_perception_data = orig_bev_data_from_all_cav[self.ids].clone()  # Clone to avoid reference issues
        #         # Select random delay frames from [1, 2, 4]
        #         self.delay_frames = random.choice([0, 1, 2, 4])
        #         self.counter = 0  # Reset counter
        #         # print(f"Stored data for IDs {self.ids} with delay of {self.delay_frames} frames.")
        
        # # Increment counter if a delay is in progress
        # if self.delay_frames>0:
        #     if self.counter < self.delay_frames and self.previous_perception_data is not None:
        #         self.counter += 1
        #     # print(f"Counter: {self.counter}/{self.delay_frames}")
        
        # # Replace data if counter reaches delay_frames
        # if self.delay_frames>0:
        #     if self.counter == self.delay_frames and self.previous_perception_data is not None:
        #         for idx, stored_id in enumerate(self.ids):
        #             if stored_id < orig_bev_data_from_all_cav.shape[0]:  # Ensure ID is within bounds
        #                 orig_bev_data_from_all_cav[stored_id] = self.previous_perception_data[idx]
        #         # print(f"Replaced data for IDs {self.ids} after {self.delay_frames} frames.")
        #         # Reset variables after replacement
        #         self.ids = []
        #         self.previous_perception_data = None
        #         self.counter = 0
        #         self.delay_frames = 0
        


        x = orig_bev_data_from_all_cav

        #shilpa max_cav change
        # Number of records to keep
        # k = 1
        # if x.shape[0] > k:
        #     # Truncate the tensor to keep only the first k records
        #     x = x[:k]
        #     # Update record_len to reflect the new number of records
        #     record_len = torch.tensor([k], device=x.device)

        x, _ = regroup(x, record_len, self.max_cav)
        #shilpa max_cav change
        # identity_matrix = torch.eye(4)  # 4x4 identity matrix
        # transformation_matrix[0, k:] = identity_matrix
        x = self.sttf(x, transformation_matrix)
        
        x = rearrange(x, 'b l h w c -> b l c h w')

        #shilpa rebuttal loss simulation
        # x, loss_percent = self.simulate_data_loss(x, record_len)
        # file_path = '/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/coopertrim_cobevt_dyn_loss10_losses_all.txt'
        # # Check if the file exists to determine the starting frame
        # if os.path.exists(file_path):
        #             # Read the last line to get the last frame number
        #             with open(file_path, 'r') as file:
        #                 lines = file.readlines()
        #                 if lines:
        #                     last_line = lines[-1]
        #                     last_frame = int(last_line.split(',')[0].strip('()'))  # Extract the frame number
        #                     current_frame = last_frame + 1
        #                 else:
        #                     current_frame = 1  # If file is empty, start with frame 1
        # else:
        #             current_frame = 1  # If file doesn't exist, start with frame 1

        #         # Prepare the line to be written to the file
        # line_to_write = f"({current_frame},{loss_percent})\n"

        #         # Write to the file
        # with open(file_path, 'a') as file:  # 'a' mode opens the file for appending
        #             file.write(line_to_write)
        #         #    print(f"Frame {current_frame}: {percentage_selected:.2f}% of indices selected") 


        n, c, h, w = orig_bev_data_from_all_cav.shape
        #shilpa max_cav change
        # n = record_len.item()
        max_cav = x.shape[1]  # max_cav = 5 (from x.shape)
        batch_size = x.shape[0]


        selected_output_values = torch.zeros(batch_size, max_cav, selected_indices.shape[0], h, w, device=x.device) 
        for idx, value in enumerate(selected_indices):
                # Use advanced indexing to copy values
                selected_output_values[:, :, idx, :,:] = x[:, :, value, :,:].clone()

        cav_id_0_data = orig_bev_data_from_all_cav[batch_dict['ego_mat_index'][0]]  # Shape: [128, 32, 32]

        #enable for fuse auto

        # # # Step 2: Replicate cav_id=0 data across all CAVs
        replicated_data = cav_id_0_data.unsqueeze(0).expand(n, -1, -1, -1)  # Shape: [5, 128, 32, 32]
        replicated_data = replicated_data.unsqueeze(0).expand(1, -1, -1, -1, -1)  # Shape: [1, 5, 128, 32, 32]

        
        selected_output_values_k = selected_output_values[:, :n, :, :]  # Shape: [1, k, 128, 307]

        #shilpa rebuttal compression
        factor = 32  
        # mode = "lossless"  # or "lossy"
        mode = "lossy" #"lossy"  # or "lossless"
        compression_filename = f"/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/coopertrim_st_compression_stats_{str(factor)}x_lossy.csv"
        selected_output_values_k = self.compress_features(selected_output_values_k, factor=factor, csv_filename = compression_filename, mode = mode)
               
        replicated_data = replicated_data.clone()
        replicated_data[:, :, selected_indices, :, :] = selected_output_values_k[:, :, :len(selected_indices), :, :]
        replicated_data=replicated_data.squeeze(0)

        x = replicated_data
        
        # compressor
        #shilpa - to check during ablation study
        if self.compression:
            x = self.naive_compressor(x)

        # Reformat to (B, max_cav, C, H, W)
        x, mask = regroup(x, record_len, self.max_cav)
        
        x = rearrange(x, 'b l c h w -> b l h w c')
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
            3) if not self.use_roi_mask \
            else get_roi_and_cav_mask(x.shape,
                                      mask,
                                      transformation_matrix,
                                      self.discrete_ratio,
                                      self.downsample_rate)
        
      
        x = rearrange(x, 'b l h w c -> b l c h w')
        
    
        x = self.fusion_net(x, com_mask)

        # #shilpa grad cam
        # if self.frame_counter % 1 == 0:
        #     output_dir = f"/data/HangQiu/proj/AutoNetSelection/eval_visualizations/fused_frame_{self.frame_counter}"
        #     self.fax.visualize_selected_channels(
        #         x.squeeze(0), 
        #         selected_indices, 
        #         output_dir,
        #         self.frame_counter
        #     )
        # self.frame_counter += 1

        #shilpa prev feature for uncertainty improvement
        self.prev_fused_feature = x.squeeze(0).clone()

        x = x.unsqueeze(1)



        
        # dynamic head
        x = self.decoder(x)
        x = rearrange(x, 'b l c h w -> (b l) c h w')
        b = x.shape[0]
        output_dict = self.seg_head(x, b, 1)

        
        #shilpa select threshold
        return output_dict, select_threhold, percentage_selected
        # return output_dict, channel_select_probabilities, percentage_selected, state 