"""
Implementation of V2VNet Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from opencood.models.sub_modules.torch_transformation_utils import \
    get_discretized_transformation_matrix, get_transformation_matrix, \
    warp_affine, get_rotated_roi
from opencood.models.sub_modules.convgru import ConvGRU

#shilpa channel select adapt
from opencood.models.sub_modules.channel_select_attention import CrossAttentionMaskPredictor, CrossAttentionMaskPredictorAdaptive
#shilpa channel select adapt SA
# from opencood.models.sub_modules.channel_select_self_attention import SelfAttentionMaskPredictor
import os
import numpy as np

#shilpa epsilon greedy
import random



class PixelWeightedFusionSoftmax(nn.Module):
    def __init__(self, channel):
        super(PixelWeightedFusionSoftmax, self).__init__()

        self.conv1_1 = nn.Conv2d(channel * 2, 128, kernel_size=1, stride=1,
                                 padding=0)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=0)
        # self.bn1_4 = nn.BatchNorm2d(1)

    def forward(self, x, mask=None):
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        x_1 = F.relu(self.conv1_4(x_1))
        if mask is not None:
            x_1 = x_1.masked_fill(mask == 0, -float('inf'))
        return self.softmax(x_1)


class DiscoNetFusion(nn.Module):
    def __init__(self, args):
        super(DiscoNetFusion, self).__init__()
        in_channels = args['in_channels']
        H, W = args['conv_gru']['H'], args['conv_gru']['W']
        kernel_size = args['conv_gru']['kernel_size']
        num_gru_layers = args['conv_gru']['num_layers']

        self.use_temporal_encoding = args['use_temporal_encoding']
        self.discrete_ratio = args['resolution']
        self.downsample_rate = args['downsample_rate']
        self.num_iteration = args['num_iteration']
        self.gru_flag = args['gru_flag']
        self.agg_operator = args['agg_operator']
        self.use_mask = args['use_mask']

        self.cnn = nn.Conv2d(in_channels + 1, in_channels, kernel_size=3,
                             stride=1, padding=1)
        self.msg_cnn = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3,
                                 stride=1, padding=1)
        self.conv_gru = ConvGRU(input_size=(H, W),
                                input_dim=in_channels * 2,
                                hidden_dim=[in_channels],
                                kernel_size=kernel_size,
                                num_layers=num_gru_layers,
                                batch_first=True,
                                bias=True,
                                return_all_layers=False)
        self.mlp = nn.Linear(in_channels, in_channels)
        self.pixel_weighted_fusion = PixelWeightedFusionSoftmax(in_channels)

        #shilpa channel selection
        self.first_frame = True

        #shilpa channel select adapt
        # num_channel_select = args['channel_select']['channel_dim']
        # num_spatial_select = args['channel_select']['spatial_dim']
        # self.channel_select_model = CrossAttentionMaskPredictor(num_channels=num_channel_select, spatial_dim=num_spatial_select)
        # self.channel_select_model_adaptive = CrossAttentionMaskPredictorAdaptive(num_channels=num_channel_select, spatial_dim=num_spatial_select)

    
        # shilpa learnable confidence level
        # initial_confidence_level = 50  # Initial value for the confidence level
        # self.confidence_level = nn.Parameter(torch.tensor(initial_confidence_level, dtype=torch.float32))
    

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    
       
    
    #shilpa channel selection
    # def process_features(self, ego_agent_feature, neighbor_feature, epoch, prev_fused_feature=None):
    def process_features(self, percentage_to_request, ego_agent_feature, neighbor_feature):
        """
        Process ego_agent_feature and neighbor_feature based on the given steps.

        Args:
            ego_agent_feature (torch.Tensor): Tensor of shape [N, C, H, W] (e.g., [4, 128, 32, 32]).
            neighbor_feature (torch.Tensor): Tensor of shape [N, C, H, W] (e.g., [4, 128, 32, 32]).

        Returns:
            torch.Tensor: The processed selected_data tensor of shape [N, C, H, W].
        """
        # Step 1: Compute standard deviation and get top 80% indices
        ego_agent_sample = ego_agent_feature[0]  # Shape: [C, H, W]

        # shilpa channel select
        std_dev = torch.std(ego_agent_sample, dim=(1, 2))  # Shape: [C]
        top_k = int(percentage_to_request * std_dev.numel())  # 80% of channels
        _, top_k_indices = torch.topk(std_dev, top_k)  # Indices of top 80% std-dev channels

        #shilpa random select
        # num_channels = ego_agent_sample.shape[0]  # Number of channels (C)
        # top_k = int(percentage_to_request * num_channels)  # e.g., 80% of channels
        # top_k_indices = torch.randperm(num_channels)[:top_k]  # Randomly shuffle and select top_k indices

        # Step 2: Create selected_data by repeating ego_agent_feature[0]
        n = ego_agent_feature.shape[0]  # Number of samples in the batch (N)
        selected_data = ego_agent_sample.unsqueeze(0).repeat(n, 1, 1, 1)  # Shape: [N, C, H, W]

        # Step 3: Replace top_k_indices in selected_data with corresponding values from neighbor_feature
        selected_data[:, top_k_indices, :, :] = neighbor_feature[:, top_k_indices, :, :]
        
        return selected_data#, select_threshold, percentage_selected

    def forward(self, x, record_len, pairwise_t_matrix):
        # x: (B,C,H,W)
        # record_len: (B)
        # pairwise_t_matrix: (B,L,L,4,4)
        # prior_encoding: (B,3)
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        # split x:[(L1, C, H, W), (L2, C, H, W)]
        split_x = self.regroup(x, record_len)
        # (B,L,L,2,3)
        pairwise_t_matrix = get_discretized_transformation_matrix(
            pairwise_t_matrix.reshape(-1, L, 4, 4), self.discrete_ratio,
            self.downsample_rate).reshape(B, L, L, 2, 3)
        # (B*L,L,1,H,W)
        roi_mask = get_rotated_roi((B * L, L, 1, H, W),
                                   pairwise_t_matrix.reshape(B * L * L, 2, 3))
        roi_mask = roi_mask.reshape(B, L, L, 1, H, W)

        batch_node_features = split_x

       

        # iteratively update the features for num_iteration times
        for l in range(self.num_iteration):

            batch_updated_node_features = []
            # iterate each batch
            for b in range(B):

                # number of valid agent
                N = record_len[b]
                # (N,N,4,4)
                # t_matrix[i, j]-> from i to j
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
                updated_node_features = []
                # update each node i
                for i in range(N):
                    # (N,1,H,W)
                    mask = roi_mask[b, :N, i, ...]

                    # flip the feature so the transformation is correct
                    batch_node_feature = batch_node_features[b]
                    batch_node_feature = rearrange(batch_node_feature,
                                                   'b c h w  -> b c w h')
                    batch_node_feature = torch.flip(batch_node_feature,
                                                    dims=(3,))


                    current_t_matrix = t_matrix[:, i, :, :]
                    current_t_matrix = get_transformation_matrix(
                        current_t_matrix, (H, W))

                    # (N,C,H,W)
                    neighbor_feature = warp_affine(batch_node_feature,
                                                   current_t_matrix,
                                                   (H, W))
                    # (N,C,H,W)
                    ego_agent_feature = batch_node_feature[i].unsqueeze(
                        0).repeat(N, 1, 1, 1)
                    
                    #shilpa channel selection
                    if self.first_frame:
                        percentage_to_request = 1.0
                        self.first_frame = False
                    else:
                        percentage_to_request = 0.01
                        # print("Using 1% of channels for fusion")
                        # percentage_to_request = 0.5
                        # print("Using 50% of channels for fusion")
                    neighbor_feature = self.process_features(percentage_to_request, ego_agent_feature, neighbor_feature)
                    # neighbor_feature, select_threshold, percentage_selected = self.process_features(ego_agent_feature, neighbor_feature,epoch, prev_fused_feature)
                    # Store select_threshold and percentage_selected
                   
                    
                    # (N,1,H,W)
                    if self.use_mask:
                        AgentWeight = self.pixel_weighted_fusion(
                            torch.cat([neighbor_feature, ego_agent_feature],
                                  dim=1),
                            mask)
                    else:
                        AgentWeight = self.pixel_weighted_fusion(
                            torch.cat([neighbor_feature, ego_agent_feature],
                                      dim=1))

                    # (C,H,W)
                    ego_updated_features = (
                                AgentWeight * neighbor_feature * mask).sum(0)
                    ego_updated_features = torch.flip(ego_updated_features,
                                                      dims=(2, ))
                    ego_updated_features = rearrange(ego_updated_features,
                                                     'c w h -> c h w ')

                    updated_node_features.append(
                        ego_updated_features.unsqueeze(0))
                # (N,C,H,W)
                batch_updated_node_features.append(
                    torch.cat(updated_node_features, dim=0))
            batch_node_features = batch_updated_node_features

        # (B,C,H,W)
        out = torch.cat(
            [itm[0, ...].unsqueeze(0) for itm in batch_node_features], dim=0)
        # (B,C,H,W)
        out = self.mlp(out.permute(0, 2, 3, 1))

       
        # #  File path
        # print(f"percentage_selected: {mean_percentage_selected.item():.2f}%")
        # file_path = '/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/channel_usage_disconet_st_lagrange.txt'
        # # Check if the file exists to determine the starting frame
        # if os.path.exists(file_path):
        #     # Read the last line to get the last frame number
        #     with open(file_path, 'r') as file:
        #         lines = file.readlines()
        #         # print(lines)
        #         if lines:
        #             last_line = lines[-1]
        #             last_frame = int(last_line.split(',')[0].strip('()'))  # Extract the frame number
        #             current_frame = last_frame + 1
        #         else:
        #             current_frame = 1  # If file is empty, start with frame 1
            
        #     # Prepare the line to be written to the file
        #     line_to_write = f"({current_frame},{mean_percentage_selected.item()})\n"
        #     # Write to the file
        #     with open(file_path, 'a') as file:  # 'a' mode opens the file for appending
        #         file.write(line_to_write)
        #             # print(f"Frame {current_frame}: {percentage_selected:.2f}% of indices selected") 
        # else:
        #     current_frame = 1  # If file doesn't exist, start with frame 1
        #     line_to_write = f"({current_frame},{mean_percentage_selected.item()})\n"
        #     # Write to the file
        #     with open(file_path, 'a') as file:  # 'a' mode opens the file for appending
        #         file.write(line_to_write)
        #         # print(f"Frame {current_frame}: {percentage_selected:.2f}% of indices selected") 

        # return out
        return out #, mean_select_threshold.item(), mean_percentage_selected.item()
