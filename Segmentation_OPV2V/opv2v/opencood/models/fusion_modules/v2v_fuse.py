"""
Implementation of V2VNet Fusion
"""

import torch
import torch.nn as nn
from einops import rearrange

from opencood.models.sub_modules.torch_transformation_utils import \
    get_discretized_transformation_matrix, warp_affine, get_rotated_roi,\
    get_transformation_matrix
from opencood.models.sub_modules.convgru import ConvGRU

#shilpa channel select adapt
from opencood.models.sub_modules.channel_select_attention import CrossAttentionMaskPredictor
#shilpa channel select adapt SA
# from opencood.models.sub_modules.channel_select_self_attention import SelfAttentionMaskPredictor
import os


class V2VNetFusion(nn.Module):
    def __init__(self, args):
        super(V2VNetFusion, self).__init__()
        in_channels = args['in_channels']
        H, W = args['conv_gru']['H'], args['conv_gru']['W']
        kernel_size = args['conv_gru']['kernel_size']
        num_gru_layers = args['conv_gru']['num_layers']

        self.discrete_ratio = args['resolution']
        self.downsample_rate = args['downsample_rate']

        self.num_iteration = args['num_iteration']
        self.gru_flag = args['gru_flag']
        self.agg_operator = args['agg_operator']

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
        #shilpa channel selection
        self.first_frame = True

        #shilpa channel select adapt
        # num_channel_select = args['channel_select']['channel_dim']
        # num_spatial_select = args['channel_select']['spatial_dim']
        # self.channel_select_model = CrossAttentionMaskPredictor(num_channels=num_channel_select, spatial_dim=num_spatial_select)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

     #shilpa channel selection
    def process_features(self, percentage_to_request, ego_agent_feature, neighbor_feature):
        """
        Process ego_agent_feature and neighbor_feature based on the given steps.

        Args:
            ego_agent_feature (torch.Tensor): Tensor of shape [N, C, H, W] (e.g., [4, 128, 32, 32]).
            neighbor_feature (torch.Tensor): Tensor of shape [N, C, H, W] (e.g., [4, 128, 32, 32]).

        Returns:
            torch.Tensor: The processed selected_data tensor of shape [N, C, H, W].
        """
      
        ego_agent_sample = ego_agent_feature
        #shilpa channel select
        std_dev = torch.std(ego_agent_sample, dim=(1, 2))  # Shape: [C]
        top_k = int(percentage_to_request * std_dev.numel())  # 80% of channels
        _, top_k_indices = torch.topk(std_dev, top_k)  # Indices of top 80% std-dev channels

        #shilpa random select
        # num_channels = ego_agent_sample.shape[0]  # Number of channels (C)
        # top_k = int(percentage_to_request * num_channels)  # e.g., 80% of channels
        # top_k_indices = torch.randperm(num_channels)[:top_k]  # Randomly shuffle and select top_k indices

        #shilpa channel select adapt
        # std_dev = torch.std(ego_agent_sample, dim=(1, 2))
        # std_dev = std_dev.unsqueeze(0)  
        # predicted_mask = self.channel_select_model(std_dev, ego_agent_sample.unsqueeze(0))  # Shape: [batch_size, 128]
        # #shilpa channel select adapt SA
        # # predicted_mask = self.channel_select_model(data_at_index_0.unsqueeze(0))  # Shape: [batch_size, 128]
        # top_k_indices = (predicted_mask > 0.5).float().to(ego_agent_sample.device)  # Threshold at 0.5
        # top_k_indices = top_k_indices.squeeze(0)
        # top_k_indices = top_k_indices.long()

        # total_indices = top_k_indices.numel()
        # selected_indices = top_k_indices.sum().item()  # Count of selected indices
        # percentage_selected = (selected_indices / total_indices) * 100
        # # print(f"percentage_selected: {percentage_selected:.2f}%")

        # #  File path
        # file_path = '/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/channel_usage_disconet_CA_dyn.txt'
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
        #     # Calculate the percentage of selected indices
        #     total_indices = top_k_indices.numel()
        #     selected_indices = top_k_indices.sum().item()  # Count of selected indices
        #     percentage_selected = (selected_indices / total_indices) * 100
        #     # Prepare the line to be written to the file
        #     line_to_write = f"({current_frame},{percentage_selected})\n"
        #     # Write to the file
        #     with open(file_path, 'a') as file:  # 'a' mode opens the file for appending
        #         file.write(line_to_write)
        #             # print(f"Frame {current_frame}: {percentage_selected:.2f}% of indices selected") 
        # else:
        #     current_frame = 1  # If file doesn't exist, start with frame 1
        #     # Calculate the percentage of selected indices
        #     total_indices = top_k_indices.numel()
        #     selected_indices = top_k_indices.sum().item()  # Count of selected indices
        #     percentage_selected = (selected_indices / total_indices) * 100
        #     # Prepare the line to be written to the file
        #     line_to_write = f"({current_frame},{percentage_selected})\n"
        #     # Write to the file
        #     with open(file_path, 'a') as file:  # 'a' mode opens the file for appending
        #         file.write(line_to_write)
        #             # print(f"Frame {current_frame}: {percentage_selected:.2f}% of indices selected") 
        

        # Step 2: Create selected_data by repeating ego_agent_feature[0]
        n = neighbor_feature.shape[0]  # Number of samples in the batch (N)
        selected_data = ego_agent_sample.unsqueeze(0).repeat(n, 1, 1, 1)  # Shape: [N, C, H, W]

        # Step 3: Replace top_k_indices in selected_data with corresponding values from neighbor_feature
        selected_data[:, top_k_indices, :, :] = neighbor_feature[:, top_k_indices, :, :]
        
        return selected_data

    def forward(self, x, record_len, pairwise_t_matrix, prior_encoding):
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
                    #shilpa channel selection
                    ego_feat = batch_node_feature[i]
                    if self.first_frame:
                        percentage_to_request = 1.0
                        self.first_frame = False
                    else:
                        percentage_to_request = 0.5
                        # print("Using 50% of channels for fusion")
                    neighbor_feature = self.process_features(percentage_to_request, ego_feat, neighbor_feature)

                    # (N,C,H,W)
                    ego_agent_feature = batch_node_feature[i].unsqueeze(
                        0).repeat(N, 1, 1, 1)
                    # (N,2C,H,W)
                    neighbor_feature = torch.cat(
                        [neighbor_feature, ego_agent_feature], dim=1)
                    # (N,C,H,W)
                    message = self.msg_cnn(neighbor_feature) * mask

                    # (C,H,W)
                    if self.agg_operator == "avg":
                        agg_feature = torch.mean(message, dim=0)
                    elif self.agg_operator == "max":
                        agg_feature = torch.max(message, dim=0)[0]
                    else:
                        raise ValueError("agg_operator has wrong value")
                    # (2C, H, W)
                    cat_feature = torch.cat(
                        [batch_node_feature[i, ...], agg_feature], dim=0)
                    # (C,H,W)
                    if self.gru_flag:
                        gru_out = \
                            self.conv_gru(
                                cat_feature.unsqueeze(0).unsqueeze(0))[
                                0][
                                0].squeeze(0).squeeze(0)
                    else:
                        gru_out = batch_node_feature[i, ...] + agg_feature

                    gru_out = torch.flip(gru_out,
                                         dims=(2,))
                    gru_out = rearrange(gru_out,
                                        'c w h -> c h w ')

                    updated_node_features.append(gru_out.unsqueeze(0))
                # (N,C,H,W)
                batch_updated_node_features.append(
                    torch.cat(updated_node_features, dim=0))
            batch_node_features = batch_updated_node_features
        # (B,C,H,W)
        out = torch.cat(
            [itm[0, ...].unsqueeze(0) for itm in batch_node_features], dim=0)
        # (B,H,W,C)
        out = self.mlp(out.permute(0, 2, 3, 1))

        return out
