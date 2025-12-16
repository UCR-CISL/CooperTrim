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
        
        #shilpa prev feature for uncertainty improvement
        self.prev_fused_feature = None

    def forward(self, batch_dict, epoch):
        x = batch_dict['inputs']
        b, l, m, _, _, _ = x.shape

        # shape: (B, max_cav, 4, 4)
        transformation_matrix = batch_dict['transformation_matrix']
        record_len = batch_dict['record_len']

        x = self.encoder(x)
        batch_dict.update({'features': x})
        #shilpa channel entropy
        # x = self.cvm(batch_dict)
        orig_bev_data_from_all_cav, selected_indices, select_threhold, percentage_selected = self.cvm(batch_dict, epoch, self.prev_fused_feature)
        # orig_bev_data_from_all_cav, selected_indices = self.cvm(batch_dict)
        
        x = orig_bev_data_from_all_cav
        x, _ = regroup(x, record_len, self.max_cav)
        x = self.sttf(x, transformation_matrix)
        x = rearrange(x, 'b l h w c -> b l c h w')
        n, c, h, w = orig_bev_data_from_all_cav.shape
        max_cav = x.shape[1]  # max_cav = 5 (from x.shape)
        batch_size = x.shape[0]
        # print(f"x.device: {x.device}, orig_bev_data_from_all_cav.device: {orig_bev_data_from_all_cav.device}, selected_indices.device: {selected_indices.device}")
        selected_output_values = torch.zeros(batch_size, max_cav, selected_indices.shape[0], h, w, device=x.device) 
        for idx, value in enumerate(selected_indices):
                # Use advanced indexing to copy values
                selected_output_values[:, :, idx, :,:] = x[:, :, value, :,:].clone()

        cav_id_0_data = orig_bev_data_from_all_cav[batch_dict['ego_mat_index'][0]]  # Shape: [128, 32, 32]
        # # # Step 2: Replicate cav_id=0 data across all CAVs
        replicated_data = cav_id_0_data.unsqueeze(0).expand(n, -1, -1, -1)  # Shape: [5, 128, 32, 32]
        replicated_data = replicated_data.unsqueeze(0).expand(1, -1, -1, -1, -1)  # Shape: [1, 5, 128, 32, 32]
        selected_output_values_k = selected_output_values[:, :n, :, :]  # Shape: [1, k, 128, 307]
        replicated_data = replicated_data.clone()
        replicated_data[:, :, selected_indices, :, :] = selected_output_values_k[:, :, :len(selected_indices), :, :]
        replicated_data=replicated_data.squeeze(0)
        x = replicated_data

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

        # x = rearrange(x, 'b l h w c -> b l c h w')
        # fuse all agents together to get a single bev map, b h w c
        x = self.fusion_net(x, com_mask)
       
        x = x.unsqueeze(1).permute(0, 1, 4, 2, 3)
        #shilpa prev feature for uncertainty improvement
        self.prev_fused_feature = x.squeeze(0).squeeze(0).clone()
        # dynamic head
        x = self.decoder(x)
        x = rearrange(x, 'b l c h w -> (b l) c h w')
        # L = 1 for sure in intermedaite fusion at this point
        b = x.shape[0]
        output_dict = self.seg_head(x, b, 1)

        return output_dict, select_threhold, percentage_selected
        # return output_dict


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
