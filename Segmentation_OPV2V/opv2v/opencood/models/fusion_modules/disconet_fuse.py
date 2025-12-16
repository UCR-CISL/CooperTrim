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
        num_channel_select = args['channel_select']['channel_dim']
        num_spatial_select = args['channel_select']['spatial_dim']
        self.channel_select_model = CrossAttentionMaskPredictor(num_channels=num_channel_select, spatial_dim=num_spatial_select)
        self.channel_select_model_adaptive = CrossAttentionMaskPredictorAdaptive(num_channels=num_channel_select, spatial_dim=num_spatial_select)

    
        # shilpa learnable confidence level
        initial_confidence_level = 50  # Initial value for the confidence level
        self.confidence_level = nn.Parameter(torch.tensor(initial_confidence_level, dtype=torch.float32))
    

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    
     #shilpa conformal prediction
    # def compute_conformal_uncertainty(self, reference_data, current_data, confidence_level=90):
    #     """
    #     Compute uncertainty using conformal prediction based on reference tensor.

    #     Parameters:
    #         reference_data (torch.Tensor): Reference tensor of shape (128, 32, 32).
    #         current_data (torch.Tensor): Current tensor of shape (128, 32, 32).
    #         confidence_level (int): Confidence level for quantile threshold (e.g., 90 for 90%).

    #     Returns:
    #         quantile_threshold (float): The quantile threshold for conformity scores.
    #         uncertainty_flags (torch.Tensor): Boolean tensor of shape (128) indicating uncertainty.
    #         uncertainty_intervals (list): List of tuples containing lower and upper bounds for uncertainty intervals.
    #     """
    #     # Step 1: Compute conformity scores (mean absolute difference across spatial dimensions)
    #     conformity_scores_whole = torch.abs(reference_data - current_data)
    #     conformity_scores = conformity_scores_whole.mean(dim=(1, 2))  # Shape: [128]

    #     # Step 2: Determine the quantile threshold (e.g., 90th percentile for 90% confidence)
    #     conformity_scores_np = conformity_scores.detach().cpu().numpy()
    #     quantile_threshold = np.percentile(conformity_scores_np, confidence_level)  # Convert to numpy for quantile calculation

    #     # Step 3: Inductive uncertainty quantification
    #     # Flag channels where the conformity score exceeds the quantile threshold
    #     uncertainty_flags = conformity_scores > quantile_threshold  # Shape: [128]

    #     # Step 4: Generate uncertainty intervals
    #     uncertainty_intervals = torch.zeros(current_data.shape[0], device=current_data.device)

    #     for i in range(uncertainty_intervals.shape[0]):
    #         if uncertainty_flags[i]:
    #             # Example: Create interval based on score and threshold
    #             lower_bound = conformity_scores_whole[i].min().item()
    #             upper_bound = conformity_scores_whole[i].max().item()
    #             uncertainty_intervals[i] = upper_bound - lower_bound
    #         else:
    #             uncertainty_intervals[i] = 0.0  # No interval for conforming channels

    #     # return quantile_threshold, uncertainty_flags, uncertainty_intervals
    #     return uncertainty_intervals

    def compute_conformal_uncertainty(self, reference_data, current_data, confidence_level=90):
        """
        Compute uncertainty using conformal prediction based on reference tensor.

        Parameters:
            reference_data (torch.Tensor): Reference tensor of shape (128, 32, 32).
            current_data (torch.Tensor): Current tensor of shape (128, 32, 32).
            confidence_level (int): Confidence level for quantile threshold (e.g., 90 for 90%).

        Returns:
            quantile_threshold (float): The quantile threshold for conformity scores.
            uncertainty_flags (torch.Tensor): Boolean tensor of shape (128) indicating uncertainty.
            uncertainty_intervals (list): List of tuples containing lower and upper bounds for uncertainty intervals.
        """
        # Step 1: Compute conformity scores (mean absolute difference across spatial dimensions)
        conformity_scores_whole = torch.abs(reference_data - current_data)
        conformity_scores = conformity_scores_whole.mean(dim=(1, 2))  # Shape: [128]

        # Step 2: Determine the quantile threshold (e.g., 90th percentile for 90% confidence)
        conformity_scores_np = conformity_scores.detach().cpu().numpy()

        # shilpa learnable confidence level
        # quantile_threshold = np.percentile(conformity_scores_np, confidence_level)  # Convert to numpy for quantile calculation
        quantile_threshold = np.percentile(conformity_scores_np, confidence_level.item())

        # Step 3: Inductive uncertainty quantification
        # Flag channels where the conformity score exceeds the quantile threshold
        uncertainty_flags = conformity_scores > quantile_threshold  # Shape: [128]

        # Step 4: Generate uncertainty intervals
        uncertainty_intervals = torch.zeros(current_data.shape[0], device=current_data.device)

        for i in range(uncertainty_intervals.shape[0]):
            if uncertainty_flags[i]:
                # Example: Create interval based on score and threshold
                lower_bound = conformity_scores_whole[i].min().item()
                upper_bound = conformity_scores_whole[i].max().item()
                uncertainty_intervals[i] = upper_bound - lower_bound
            else:
                uncertainty_intervals[i] = 0.0  # No interval for conforming channels

        # return quantile_threshold, uncertainty_flags, uncertainty_intervals
        return uncertainty_intervals
    
    
    #shilpa channel selection
    def process_features(self, ego_agent_feature, neighbor_feature, epoch, prev_fused_feature=None):
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

      

        #shilpa epsilon greedy
        epsilon = 0.1  # Exploration probability
        if epoch > 5 and random.random() > epsilon:  # Exploit with probability

        # if prev_fused_feature is not None:
            # uncertainty_intervals = self.compute_conformal_uncertainty(prev_fused_feature, ego_agent_sample, confidence_level=50).unsqueeze(0)
        #     # shilpa learnable confidence level
            uncertainty_intervals = self.compute_conformal_uncertainty(prev_fused_feature, ego_agent_sample, confidence_level=self.confidence_level).unsqueeze(0)
            # predicted_mask= self.channel_select_model(uncertainty_intervals, ego_agent_sample.unsqueeze(0))
            predicted_mask, select_threshold = self.channel_select_model_adaptive(uncertainty_intervals, ego_agent_sample.unsqueeze(0))
            # select_threshold = 0.5
            
            # top_k_indices = (predicted_mask > 0.5).float().to(ego_agent_sample.device)  # Threshold at 0.5
            # top_k_indices = top_k_indices.squeeze(0)


            indices = torch.where(predicted_mask[0] == 1)#.squeeze(1)
            top_k_indices = indices[0]#.squeeze(0)
            # print(random_indices.device)
            # random_indices = random_indices.long()
            total_indices = predicted_mask.shape[1]
            selected_indices = top_k_indices.shape[0]  # Count of selected indices
            percentage_selected = (selected_indices / total_indices) * 100
            # print(f"percentage_selected: {(percentage_selected):.2f}%")
    
        else:
        #     uncertainty_intervals = torch.zeros((ego_agent_sample.shape[0]), device=ego_agent_sample.device).unsqueeze(0)         
            num_channels = ego_agent_sample.shape[0]  # Number of channels (C)
            top_k = int(1.0 * num_channels)  # e.g., 80% of channels
            top_k_indices = torch.arange(num_channels, device=ego_agent_feature.device)[:top_k] 
            select_threshold = 1.0
            percentage_selected = 100.0
            # print(f"percentage_selected: {(percentage_selected):.2f}%")
        
        

        # Step 2: Create selected_data by repeating ego_agent_feature[0]
        n = ego_agent_feature.shape[0]  # Number of samples in the batch (N)
        selected_data = ego_agent_sample.unsqueeze(0).repeat(n, 1, 1, 1)  # Shape: [N, C, H, W]

        # Step 3: Replace top_k_indices in selected_data with corresponding values from neighbor_feature
        selected_data[:, top_k_indices, :, :] = neighbor_feature[:, top_k_indices, :, :]
        
        return selected_data, select_threshold, percentage_selected

    def forward(self, x, record_len, pairwise_t_matrix, epoch, prior_encoding=None, prev_fused_feature=None):
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

        #shilpa channel selection
        # Initialize arrays to store select_threshold and percentage_selected
        all_select_thresholds = []
        all_percentage_selected = []

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
                    # if self.first_frame:
                    #     percentage_to_request = 1.0
                    #     self.first_frame = False
                    # else:
                    #     percentage_to_request = 0.05
                        # print("Using 50% of channels for fusion")
                    # neighbor_feature = self.process_features(percentage_to_request, ego_agent_feature, neighbor_feature)
                    neighbor_feature, select_threshold, percentage_selected = self.process_features(ego_agent_feature, neighbor_feature,epoch, prev_fused_feature)
                    # Store select_threshold and percentage_selected
                    all_select_thresholds.append(select_threshold)
                    all_percentage_selected.append(percentage_selected)
                    
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

        #shilpa channel selection
        # Calculate mean of select_threshold and percentage_selected
        mean_select_threshold = torch.mean(torch.tensor(all_select_thresholds))
        mean_percentage_selected = torch.mean(torch.tensor(all_percentage_selected))

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
        return out, mean_select_threshold.item(), mean_percentage_selected.item()
