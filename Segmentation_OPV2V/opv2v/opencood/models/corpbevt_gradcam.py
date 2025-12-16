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

#shilpa entropy
# from scipy.stats import entropy

#shilpa gradcam
# from torchcam.methods import GradCAM
# from torchcam.utils import overlay_mask
# import cv2



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
        #shilpa gradcam
        # Initialization code remains unchanged...
        # self.gradcam = GradCAM(model=self.fusion_net, target_layer='target_layer_name')  # Replace with the actual target layer name

    def forward(self, batch_dict, ppo_agent=None):
        x = batch_dict['inputs']
        b, l, m, _, _, _ = x.shape

        # shape: (B, max_cav, 4, 4)
        transformation_matrix = batch_dict['transformation_matrix']
        record_len = batch_dict['record_len']  # record_len is the number of CAVs in the scene

        x = self.encoder(x)
        batch_dict.update({'features': x})
      
        #shilpa select threshold
        #shilpa RL
        orig_bev_data_from_all_cav, selected_indices, select_threhold, percentage_selected = self.fax(batch_dict, self.prev_fused_feature)
        # orig_bev_data_from_all_cav, selected_indices, channel_select_probabilities, percentage_selected, state = self.fax(batch_dict, ppo_agent=ppo_agent)
        

        # orig_bev_data_from_all_cav, selected_indices = self.fax(batch_dict)

        # #shilpa request in bits
        # # Convert each index to 7-bit binary and concatenate into one binary string
        # binary_representation = ''.join(format(idx.item(), '07b') for idx in selected_indices)
        # # Convert the binary string into a single integer
        # combined_number = int(binary_representation, 2)
        # #retrieve back selected indices at cav
        # # Number of indices (length of original tensor)
        # num_indices = len(selected_indices)
        # # Extract 7 bits at a time and convert back to integers
        # selected_indices = torch.tensor([
        #     int(binary_representation[i:i+7], 2) for i in range(0, num_indices * 7, 7)
        # ])

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