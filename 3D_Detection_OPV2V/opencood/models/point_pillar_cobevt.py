import torch
import torch.nn as nn
from einops import rearrange, repeat

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.swap_fusion_modules import \
    SwapFusionEncoder
from opencood.models.fuse_modules.fuse_utils import regroup
#shilpa autonet
import numpy as np
from opencood.models.sub_modules.channel_select_attention import CrossAttentionMaskPredictorAdaptive
import os
import random


class PointPillarCoBEVT(nn.Module):
    def __init__(self, args):
        super(PointPillarCoBEVT, self).__init__()

        self.max_cav = args['max_cav']
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.fusion_net = SwapFusionEncoder(args['fax_fusion'])

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()

        #shilpa autonet
        #shilpa channel select adapt
        num_channel_select = 256
        num_spatial_select_h = 96 #48 #config['channel_select']['spatial_dim']
        num_spatial_select_w = 352 #176
        self.channel_select_model = CrossAttentionMaskPredictorAdaptive(num_channels=num_channel_select, spatial_dim_h=num_spatial_select_h, spatial_dim_w=num_spatial_select_w)

        #shilpa autonet
        self.prev_fused_feature = None
        # shilpa learnable confidence level
        initial_confidence_level = 50  # Initial value for the confidence level
        self.confidence_level = nn.Parameter(torch.tensor(initial_confidence_level, dtype=torch.float32))


    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    #shilpa autonet
    def divide_batches(self, spatial_features_2d, record_len):
        """
        Divide spatial_features_2d into batches based on record_len.

        Args:
            spatial_features_2d (torch.Tensor): Tensor of size (n, c, h, w).
            record_len (list): List of integers with batch sizes. Sum of record_len = n.

        Returns:
            list: List of divided batches, where each batch corresponds to the data for a single record_len.
        """
        divided_batches = []
        start_idx = 0

        for batch_size in record_len:
            # Extract the batch slice based on record_len
            batch_data = spatial_features_2d[start_idx:start_idx + batch_size]#.to(spatial_features_2d.device)
            divided_batches.append(batch_data)
            start_idx += batch_size

        return divided_batches

    #shilpa autonet
    #shilpa conformal prediction
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

    #shilpa autonet
    def create_request(self, divided_batches, channel_select_model, epoch, prev_fused_feature=None, validation=False):
        """
        Process divided_batches to integrate orig_bev_data_from_all_cav and select BEV points.

        Args:
            divided_batches (list): List of tensors, each of shape (l, c, h, w), where l is the batch size.
            record_len (list): List of integers representing batch sizes.
            channel_select_model (callable): Model to predict mask for channel selection.
            prev_avg_entropy (float, optional): Previous average entropy for adaptive selection.

        Returns:
            list: Processed batches with selected BEV data integrated.
        """
        processed_batches = []
        for i, batch_data in enumerate(divided_batches):
            # Orig BEV data from all CAVs
            orig_bev_data_from_all_cav = batch_data  # Assuming batch_data is the input BEV data

            # Process the first element (ego data) of the batch
            data_at_index_0 = orig_bev_data_from_all_cav[0]  # Shape: (c, h, w)
            dim_len, height, width = data_at_index_0.shape  # Extract dimensions

            num_spatial_indices = data_at_index_0.shape[0]  # Number of channels
            random_indices = None
            select_threshold = None
            percentage_selected = None

            # Adaptive selection logic
            # # if prev_avg_entropy is not None:            
            # std_dev = data_at_index_0.std(dim=(1, 2))  # Compute standard deviation across spatial dimensions
            # std_dev = std_dev.unsqueeze(0)  # Add batch dimension for model input
            # # Predict mask using the channel selection model
            # predicted_mask, select_threshold = channel_select_model(std_dev, data_at_index_0.unsqueeze(0))
            # random_indices = predicted_mask.squeeze(0).long()  # Convert probabilities to binary mask
            # total_indices = random_indices.numel()
            # selected_indices = random_indices.sum().item()  # Count of selected indices
            # percentage_selected = (selected_indices / total_indices) * 100
            # print(f"Batch {i}: percentage_selected = {percentage_selected:.2f}%")
            # # else:
            # #     # Default selection logic (select all data)
            # #     percentage_data_to_request = 1.0  # Request 100% of data
            # #     num_random_indices = int(percentage_data_to_request * num_spatial_indices)
            # #     random_indices = torch.arange(num_spatial_indices, device=data_at_index_0.device)[:num_random_indices]
            # #     prev_avg_entropy = 1.0  # Initialize entropy
            # #     percentage_selected = 100.0
            # #     print(f"Batch {i}: percentage_selected = {percentage_selected:.2f}%")

            epsilon = 0.1  # Exploration probability
            if (epoch >= 5 and random.random() > epsilon and prev_fused_feature is not None) or validation==True:
            # if epoch > 2 and random.random() > epsilon:  # Exploit with probability
            #         # Check if prev_fused_feature is provided
            #         if prev_fused_feature is not None:
                        # uncertainty_intervals = self.compute_conformal_uncertainty(prev_fused_feature, data_at_index_0, confidence_level=50).unsqueeze(0)
                    #     # shilpa learnable confidence level
                    uncertainty_intervals = self.compute_conformal_uncertainty(prev_fused_feature, data_at_index_0, confidence_level=self.confidence_level).unsqueeze(0)
                        

                    predicted_mask, select_threshold = self.channel_select_model(uncertainty_intervals, data_at_index_0.unsqueeze(0)) 
                    indices = torch.where(predicted_mask[0] == 1)#.squeeze(1)
                    random_indices = indices[0]#.squeeze(0)

                    total_indices = predicted_mask.shape[1]
                    selected_indices = random_indices.shape[0]  # Count of selected indices
                    percentage_selected = (selected_indices / total_indices) * 100
                    print(f"percentage_selected: {(percentage_selected):.2f}%")

                    # #    print(f"Frame {current_frame}: {percentage_selected:.2f}% of indices selected")          
            else:
                    percentage_data_to_request = 1.0
                    num_random_indices = int(percentage_data_to_request * num_spatial_indices)  # Compute 30% of total indices
                    #shilpa Transmission 1 - this data is transmitted from ego to CAV for request
                    # random_indices = torch.randperm(num_spatial_indices, device=flattened_data.device)[:num_random_indices]  # Random 30% indices
                    random_indices = torch.arange(num_spatial_indices, device=data_at_index_0.device)[:num_random_indices]
                    self.prev_avg_entropy = 1
                    percentage_selected = 100.0
                    print(f"percentage_selected: {(percentage_selected):.2f}%")

            # Integrate processed data into the batch
            processed_batch = {
                "orig_bev_data_from_all_cav": orig_bev_data_from_all_cav,#.to(divided_batches[0].device),
                "random_indices": random_indices, #.to(divided_batches[0].device),
                "select_threshold": select_threshold, #.to(divided_batches[0].device),
                "percentage_selected": percentage_selected,#.to(divided_batches[0].device),
            }
            processed_batches.append(processed_batch)

        return processed_batches
    
    #shilpa autonet
    def create_response(self, processed_batches):
        """
        Integrates processed_batches into a new tensor with transformations and replication.

        Args:
            processed_batches (list): List of processed batches containing selected indices and BEV data.
            record_len (torch.Tensor): Tensor representing the number of records in each batch.
            max_cav (int): Maximum number of CAVs to consider.
            transformation_matrix (torch.Tensor): Transformation matrix for spatial alignment.
            sttf_function (callable): Function to apply spatial transformation.

        Returns:
            torch.Tensor: Integrated tensor after processing.
        """
        batch_size = len(processed_batches)
        integrated_tensor = None  # Placeholder for the final integrated tensor
        selected_output_values =[]
        for batch_idx, processed_batch in enumerate(processed_batches):
            orig_bev_data_from_all_cav = processed_batch["orig_bev_data_from_all_cav"]#.to(processed_batch["orig_bev_data_from_all_cav"].device)
            selected_indices = processed_batch["random_indices"]

            # # Step 1: Regroup data
            # x, _ = regroup(orig_bev_data_from_all_cav, record_len, max_cav)

            # # Step 3: Rearrange dimensions
            # x = rearrange(x, 'b l h w c -> b l c h w')

            # Extract dimensions
            n, c, h, w = orig_bev_data_from_all_cav.shape
            # batch_size = processed_batches.shape[0]

            # Step 4: Select output values based on selected indices
            selected_output_values.append(torch.zeros(
                n, selected_indices.shape[0], h, w #, device=processed_batches.device
            ))
            for idx, value in enumerate(selected_indices):
                selected_output_values[-1][:, idx, :, :] = orig_bev_data_from_all_cav[:, value, :, :].clone()

        return selected_output_values
        
    def accumulate_features_at_ego(self, processed_batches, selected_output_values, target_device ) :
        batch_size = len(processed_batches)
        integrated_tensor = None  # Placeholder for the final integrated tensor
        for batch_idx, processed_batch in enumerate(processed_batches):
            orig_bev_data_from_all_cav = processed_batch["orig_bev_data_from_all_cav"]
            selected_indices = processed_batch["random_indices"].to(target_device)
            # Step 5: Replicate ego data across all CAVs
            cav_id_0_data = orig_bev_data_from_all_cav[0]
            n, c, h, w = orig_bev_data_from_all_cav.shape
            
            replicated_data = cav_id_0_data.unsqueeze(0).expand(n, -1, -1, -1)  # Shape: [n, c, h, w]
            # replicated_data = replicated_data.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # Shape: [1, n, c, h, w]
            # Step 6: Replace selected indices in replicated data
            selected_output_values_k = selected_output_values[batch_idx][:n, :, :, :].to(target_device)  # Select specific values
            replicated_data = replicated_data.clone().to(target_device)
            replicated_data[:, selected_indices, :, :] = selected_output_values_k[:, :len(selected_indices), :, :]
            # replicated_data = replicated_data.squeeze(0)
            # Update integrated tensor
            if integrated_tensor is None:
                integrated_tensor = replicated_data#.unsqueeze(0)
            else:
                integrated_tensor = torch.cat((integrated_tensor, replicated_data), dim=0) #torch.cat((integrated_tensor, replicated_data.unsqueeze(0)), dim=0)

        return integrated_tensor

    def forward(self, data_dict, epoch, validation=False):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        #shilpa autonet
        # print(f"spatial_features_2d shape = {spatial_features_2d.shape}")
        target_device = spatial_features_2d.device
        # Divide spatial_features_2d into batches
        divided_batches = self.divide_batches(spatial_features_2d, record_len)
        # Process divided_batches   
        processed_batches = self.create_request(divided_batches, self.channel_select_model, epoch, self.prev_fused_feature, validation=validation)
        selected_output_values = self.create_response(processed_batches)
        reconstructed_data_at_ego = self.accumulate_features_at_ego( processed_batches, selected_output_values, target_device )
        spatial_features_2d = reconstructed_data_at_ego.clone()
        # print(f"spatial_features_2d shape after reconstruction= {spatial_features_2d.shape}")

    
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)


        # N, C, H, W -> B,  L, C, H, W
        regroup_feature, mask = regroup(spatial_features_2d,
                                        record_len,
                                        self.max_cav)
        com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        com_mask = repeat(com_mask,
                          'b h w c l -> b (h new_h) (w new_w) c l',
                          new_h=regroup_feature.shape[3],
                          new_w=regroup_feature.shape[4])

        fused_feature = self.fusion_net(regroup_feature, com_mask)
        #shilpa autonet
        self.prev_fused_feature = fused_feature[0]  # Store ego fused feature for next iteration

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm}

        #shilpa autonet
        # Initialize empty lists to store the values
        select_thresholds = []
        percentage_selected = []

        # Iterate through the list and extract values
        for batch in processed_batches:
            select_thresholds.append(batch["select_threshold"])
            percentage_selected.append(batch["percentage_selected"])
        return output_dict, select_thresholds, percentage_selected