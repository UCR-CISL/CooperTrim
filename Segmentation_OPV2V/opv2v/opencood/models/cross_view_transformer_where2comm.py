import torch.nn as nn

# from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
# from opencood.models.fusion_modules.where2comm_fuse import Where2comm
# from opencood.models.sub_modules.downsample_conv import DownsampleConv
# from opencood.models.sub_modules.naive_compress import NaiveCompressor
# from opencood.models.sub_modules.pillar_vfe import PillarVFE
# from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter

from einops import rearrange
from opencood.models.sub_modules.cvt_modules import CrossViewModule
from opencood.models.backbones.resnet_ms import ResnetEncoder
from opencood.models.sub_modules.naive_decoder import NaiveDecoder
from opencood.models.fusion_modules.where2comm_fuse import Where2comm
from opencood.models.sub_modules.bev_seg_head import BevSegHead

import torch.nn.functional as F
import torch



class CrossViewTransformerWhere2comm(nn.Module):
    def __init__(self, config):
        super(CrossViewTransformerWhere2comm, self).__init__()
        # self.max_cav = args['max_cav']
        # # Pillar VFE
        # self.pillar_vfe = PillarVFE(args['pillar_vfe'],
        #                             num_point_features=4,
        #                             voxel_size=args['voxel_size'],
        #                             point_cloud_range=args['lidar_range'])
        # self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        # self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        # # Used to down-sample the feature map for efficient computation
        # if 'shrink_header' in args:
        #     self.shrink_flag = True
        #     self.shrink_conv = DownsampleConv(args['shrink_header'])
        # else:
        #     self.shrink_flag = False

        # if args['compression']:
        #     self.compression = True
        #     self.naive_compressor = NaiveCompressor(256, args['compression'])
        # else:
        #     self.compression = False

        # self.fusion_net = Where2comm(args['where2comm_fusion'])
        # self.multi_scale = args['where2comm_fusion']['multi_scale']

        # self.cls_head = nn.Conv2d(args['head_dim'], args['anchor_number'], kernel_size=1)
        # self.reg_head = nn.Conv2d(args['head_dim'], 7 * args['anchor_number'], kernel_size=1)

        # if args['backbone_fix']:
        #     self.backbone_fix()

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

        # spatial fusion
        self.fusion_net = Where2comm(config['where2comm_fusion'])

        # decoder params
        decoder_params = config['decoder']
        # decoder for dynamic and static differet
        self.decoder = NaiveDecoder(decoder_params)

        self.target = config['target']
        self.seg_head = BevSegHead(self.target,
                                   config['seg_head_dim'],
                                   config['output_class'])


        

    # def backbone_fix(self):
    #     """
    #     Fix the parameters of backbone during finetune on timedelay.
    #     """

    #     for p in self.pillar_vfe.parameters():
    #         p.requires_grad = False

    #     for p in self.scatter.parameters():
    #         p.requires_grad = False

    #     for p in self.backbone.parameters():
    #         p.requires_grad = False

    #     if self.compression:
    #         for p in self.naive_compressor.parameters():
    #             p.requires_grad = False
    #     if self.shrink_flag:
    #         for p in self.shrink_conv.parameters():
    #             p.requires_grad = False

    #     for p in self.cls_head.parameters():
    #         p.requires_grad = False
    #     for p in self.reg_head.parameters():
    #         p.requires_grad = False

    # def forward(self, data_dict):
    #     voxel_features = data_dict['processed_lidar']['voxel_features']
    #     voxel_coords = data_dict['processed_lidar']['voxel_coords']
    #     voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
    #     record_len = data_dict['record_len']
    #     pairwise_t_matrix = data_dict['pairwise_t_matrix']

    #     batch_dict = {'voxel_features': voxel_features,
    #                   'voxel_coords': voxel_coords,
    #                   'voxel_num_points': voxel_num_points,
    #                   'record_len': record_len}
    #     # n, 4 -> n, c
    #     batch_dict = self.pillar_vfe(batch_dict)
    #     # n, c -> N, C, H, W
    #     batch_dict = self.scatter(batch_dict)
    #     batch_dict = self.backbone(batch_dict)

    #     # N, C, H', W': [N, 256, 48, 176]
    #     spatial_features_2d = batch_dict['spatial_features_2d']
    #     # Down-sample feature to reduce memory
    #     if self.shrink_flag:
    #         spatial_features_2d = self.shrink_conv(spatial_features_2d)

    #     psm_single = self.cls_head(spatial_features_2d)

    #     # Compressor
    #     if self.compression:
    #         # The ego feature is also compressed
    #         spatial_features_2d = self.naive_compressor(spatial_features_2d)

    #     if self.multi_scale:
    #         # Bypass communication cost, communicate at high resolution, neither shrink nor compress
    #         fused_feature, communication_rates = self.fusion_net(batch_dict['spatial_features'],
    #                                                              psm_single,
    #                                                              record_len,
    #                                                              pairwise_t_matrix,
    #                                                              self.backbone)
    #         if self.shrink_flag:
    #             fused_feature = self.shrink_conv(fused_feature)
    #     else:
    #         fused_feature, communication_rates = self.fusion_net(spatial_features_2d,
    #                                                              psm_single,
    #                                                              record_len,
    #                                                              pairwise_t_matrix)

    #     psm = self.cls_head(fused_feature)
    #     rm = self.reg_head(fused_feature)

    #     output_dict = {'psm': psm, 'rm': rm, 'com': communication_rates}
    #     return output_dict

    def forward(self, batch_dict):
        x = batch_dict['inputs']
        b, l, m, _, _, _ = x.shape

        # shape: (B, max_cav, 4, 4)
        pairwise_t_matrix = batch_dict['pairwise_t_matrix']
        record_len = batch_dict['record_len']

        x = self.encoder(x)
        batch_dict.update({'features': x})
        x = self.cvm(batch_dict)

        # B*L, C, H, W
        x = x.squeeze(1)

        #CooperTrim where2comm
        # pp =x[0].unsqueeze(0)
        # pp = self.decoder(pp.unsqueeze(0))
        # pp = rearrange(pp, 'b l c h w -> (b l) c h w')
        # output_single = self.seg_head(pp, pp.shape[0], 1)
        # # score_matrix = output_single.reshape(pp.shape[0], pp.shape[1], pp.shape[2], pp.shape[3])

        # seg_confidence = output_single['dynamic_seg'] if self.target == 'dynamic' else output_single['static_seg']
        # seg_confidence = seg_confidence.squeeze(1)  # Shape: [1, 2, 256, 256]
        # # Step 2: Downsample spatial dimensions from 256x256 to 32x32
        # # Using adaptive average pooling to reduce spatial size
        # downsampled = F.adaptive_avg_pool2d(seg_confidence, output_size=(32, 32))
        # # Shape: [1, 2, 32, 32]
        # # Step 3: Handle channel mismatch (2 to 128)
        # # Option: Replicate the 2 channels to fill 128 (or map them)
        # # Since 128 / 2 = 64, repeat each channel 64 times
        # repeated_channels = downsampled.repeat(1, 64, 1, 1)  # Shape: [1, 128, 32, 32]
        # # Note: If 128 is not divisible by 2, you may need to trim or pad
        # # Resulting tensor
        # confidence_tensor = repeated_channels

        confidence_tensor = torch.zeros((record_len, x.shape[1], x.shape[2], x.shape[3]), device=x.device)
        for i in range(record_len.item()):
            pp =x[i].unsqueeze(0)
            pp = self.decoder(pp.unsqueeze(0))
            pp = rearrange(pp, 'b l c h w -> (b l) c h w')
            output_single = self.seg_head(pp, pp.shape[0], 1)
            # score_matrix = output_single.reshape(pp.shape[0], pp.shape[1], pp.shape[2], pp.shape[3])

            seg_confidence = output_single['dynamic_seg'] if self.target == 'dynamic' else output_single['static_seg']
            seg_confidence = seg_confidence.squeeze(1)  # Shape: [1, 2, 256, 256]
            # Step 2: Downsample spatial dimensions from 256x256 to 32x32
            # Using adaptive average pooling to reduce spatial size
            downsampled = F.adaptive_avg_pool2d(seg_confidence, output_size=(32, 32))
            # Shape: [1, 2, 32, 32]
            # Step 3: Handle channel mismatch (2 to 128)
            # Option: Replicate the 2 channels to fill 128 (or map them)
            # Since 128 / 2 = 64, repeat each channel 64 times
            if self.target == 'dynamic':
                repeated_channels = downsampled.repeat(1, 64, 1, 1)
            else:
                repeated_channels = downsampled.repeat(1, 43, 1, 1)
                repeated_channels = repeated_channels[:, :-1, :, :]
            # Resulting tensor
            conf = repeated_channels
            confidence_tensor[i] = conf
        x, communication_rate = self.fusion_net(x, confidence_tensor, record_len, pairwise_t_matrix, None)

        # # Reformat to (B, max_cav, C, H, W)
        # x, mask = regroup(x, record_len, self.max_cav)
        # # perform feature spatial transformation,  B, max_cav, H, W, C
        # x = self.sttf(x, transformation_matrix)
        # com_mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(
        #     3) if not self.use_roi_mask \
        #     else get_roi_and_cav_mask(x.shape,
        #                               mask,
        #                               transformation_matrix,
        #                               self.discrete_ratio,
        #                               self.downsample_rate)

        # # fuse all agents together to get a single bev map, b h w c
        # if self.target == 'dynamic':
        #     x, communication_rate = self.fusion_net(x, confidence_tensor, record_len, pairwise_t_matrix, None)
        #     # x, communication_rate = self.fusion_net(x, output_single['dynamic_seg'], record_len, pairwise_t_matrix, None)
        # else:
        #     x, communication_rate = self.fusion_net(x, confidence_tensor, record_len, pairwise_t_matrix, None)
        #     # x, communication_rate = self.fusion_net(x, output_single['static_seg'], record_len, pairwise_t_matrix, None)
        x = x.unsqueeze(1)#.permute(0, 1, 4, 2, 3)

        # dynamic head
        x = self.decoder(x)
        x = rearrange(x, 'b l c h w -> (b l) c h w')
        # L = 1 for sure in intermedaite fusion at this point
        b = x.shape[0]
        output_dict = self.seg_head(x, b, 1)

        return output_dict, communication_rate