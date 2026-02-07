"""
Implementation of Brady Zhou's cross view transformer
"""
import torch
import torch.nn as nn
from einops import rearrange
from opencood.models.sub_modules.cvt_modules import CrossViewModule
from opencood.models.backbones.resnet_ms import ResnetEncoder
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.base_transformer import BaseTransformer
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    get_transformation_matrix, warp_affine, get_roi_and_cav_mask, \
    get_discretized_transformation_matrix
from opencood.models.sub_modules.bev_seg_head import BevSegHead

#CooperTrim compression
import zlib
import csv
import os
import numpy as np

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
        x = torch.flip(x, dims=(4, ))
        x = rearrange(x, 'b l c w h -> b l h w c')

        return x


class CrossViewTransformerAttFuse(nn.Module):
    def __init__(self, config):
        super(CrossViewTransformerAttFuse, self).__init__()
        self.max_cav = config['max_cav']
        # encoder params
        self.encoder = ResnetEncoder(config['encoder'])

        # cvm params
        cvm_params = config['cvm']
        cvm_params['backbone_output_shape'] = self.encoder.output_shapes
        self.cvm = CrossViewModule(cvm_params)

        # spatial feature transform module
        self.downsample_rate = config['sttf']['downsample_rate']
        self.discrete_ratio = config['sttf']['resolution']
        self.use_roi_mask = config['sttf']['use_roi_mask']
        self.sttf = STTF(config['sttf'])

        # spatial fusion
        self.fusion_net = BaseTransformer(config['base_transformer'])

        # decoder params
        decoder_params = config['decoder']
        # decoder for dynamic and static differet
        self.decoder = NaiveDecoder(decoder_params)

        self.target = config['target']
        self.seg_head = BevSegHead(self.target,
                                   config['seg_head_dim'],
                                   config['output_class'])

    #CooperTrim rebuttal compression
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



    def forward(self, batch_dict):
        x = batch_dict['inputs']
        b, l, m, _, _, _ = x.shape

        # shape: (B, max_cav, 4, 4)
        transformation_matrix = batch_dict['transformation_matrix']
        record_len = batch_dict['record_len']

        x = self.encoder(x)
        batch_dict.update({'features': x})
        x = self.cvm(batch_dict)

        # B*L, C, H, W
        x = x.squeeze(1)

        #CooperTrim rebuttal compression
        factor = 32  
        # mode = "lossless"  # or "lossy"
        mode = "lossy" #"lossy"  # or "lossless"
        compression_filename = f"/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/attfuse_st_compression_stats_{str(factor)}x_lossy.csv"
        x = self.compress_features(x, factor=factor, csv_filename = compression_filename, mode = mode)
         
        # Reformat to (B, max_cav, C, H, W)
        x, mask = regroup(x, record_len, self.max_cav)
        # perform feature spatial transformation,  B, max_cav, H, W, C
        x = self.sttf(x, transformation_matrix)
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
            3) if not self.use_roi_mask \
            else get_roi_and_cav_mask(x.shape,
                                      mask,
                                      transformation_matrix,
                                      self.discrete_ratio,
                                      self.downsample_rate)

        # fuse all agents together to get a single bev map, b h w c
        x = self.fusion_net(x, com_mask)
        x = x.unsqueeze(1).permute(0, 1, 4, 2, 3)

        # dynamic head
        x = self.decoder(x)
        x = rearrange(x, 'b l c h w -> (b l) c h w')
        # L = 1 for sure in intermedaite fusion at this point
        b = x.shape[0]
        output_dict = self.seg_head(x, b, 1)

        return output_dict


if __name__ == '__main__':
    import os
    import torch
    from opencood.hypes_yaml.yaml_utils import load_yaml

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    test_data = torch.rand(1, 1, 4, 512, 512, 3)
    test_data = test_data.cuda()

    extrinsic = torch.rand(1, 1, 4, 4, 4)
    intrinsic = torch.rand(1, 1, 4, 3, 3)

    extrinsic = extrinsic.cuda()
    intrinsic = intrinsic.cuda()

    params = load_yaml('../hypes_yaml/opcamera/cvt.yaml')

    model = CrossViewTransformerAttFuse(params['model']['args'])
    model = model.cuda()
    while True:
        output = model({'inputs': test_data,
                        'extrinsic': extrinsic,
                        'intrinsic': intrinsic})
        print('test_passed')
