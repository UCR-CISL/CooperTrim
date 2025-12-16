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

#shilpa lidar segment
# from PIL import Image
# import numpy as np


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

    #shilpa lidar segment
    # def voxel_to_image(self, voxel_data, output_path="image_generated.png"):
    #     """
    #     Convert voxel feature data to an image and save it.

    #     Args:
    #         voxel_data (np.ndarray): Voxel data of shape (N, 32, 4), where:
    #             - N: Number of voxels
    #             - 32: Number of rows (rings)
    #             - 4: Features [x, y, z, intensity]
    #         output_path (str): Path to save the generated image.

    #     Returns:
    #         None
    #     """
    #     # Ensure voxel_data is on the CPU
    #     if voxel_data.is_cuda:
    #         voxel_data = voxel_data.cpu()

    #     # Convert to NumPy array
    #     voxel_data = voxel_data.numpy()
    #     # Initialize an empty image with 32 rows and 2048 columns
    #     img = np.zeros((256,256, 3), dtype=np.uint8)

    #     # Iterate over each voxel
    #     for voxel in voxel_data:
    #         # Extract x, y, z, and intensity for the current voxel
    #         x, y, z, intensity = voxel[:, 0], voxel[:, 1], voxel[:, 2], voxel[:, 3]

    #         # Compute range
    #         epsilon = 1e-6
    #         rang = np.sqrt(x**2 + y**2 + z**2) + epsilon

    #         # Compute row and column indices for projection
    #         fov_up = 0.392  # Upward field of view
    #         fov_down = -0.392  # Downward field of view
    #         row_scale = 256  # Number of rows
    #         col_scale = 256  # Number of columns (scan mode)

    #         u = (row_scale * (-((np.arcsin(z / rang) + fov_down) / (fov_up - fov_down)))).astype(np.int32)
    #         v = (col_scale * (0.5 * ((np.arctan2(y, x) / np.pi) + 1))).astype(np.int32)

    #         # Clip indices to valid ranges
    #         u = np.clip(u, 0, 255)
    #         v = np.clip(v, 0, 255)

    #         # Normalize intensity and range for visualization
    #         intensity_normalized = (intensity / np.max(intensity) * 255).astype(np.uint8)
    #         range_normalized = (rang / np.max(rang) * 255).astype(np.uint8)

    #         # Project intensity and range into the image
    #         for i in range(len(u)):
    #             img[u[i], v[i], 0] = intensity_normalized[i]  # Red channel: Intensity
    #             img[u[i], v[i], 1] = range_normalized[i]      # Green channel: Range

    #     # Save the image
    #     img_pil = Image.fromarray(img)
    #     img_pil.save(output_path)
    #     print(f"Image saved to {output_path}")

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']

        #shilpa lidar segment
        # self.voxel_to_image(voxel_features, "image_generated.png")

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

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict
