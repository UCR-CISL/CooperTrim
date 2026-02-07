import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List

#CooperTrim entropy
import numpy as np
#CooperTrim channel select adapt
from opencood.models.sub_modules.channel_select_attention import CrossAttentionMaskPredictor, CrossAttentionMaskPredictorAdaptive
# from opencood.models.sub_modules.channel_select_attention_ppo import CrossAttentionMaskPredictor
import os

#CooperTrim channel select adapt SA
# from opencood.models.sub_modules.channel_select_self_attention import SelfAttentionMaskPredictor
# import os

#CooperTrim grad cam
from opencood.models.sub_modules.inference_grad_cam import process_feature_visualization

#CooperTrim epsilon greedy
import random

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]


class BEVEmbedding(nn.Module):
    def __init__(
            self,
            dim: int,
            sigma: int,
            bev_height: int,
            bev_width: int,
            h_meters: int,
            w_meters: int,
            offset: int,
            upsample_scales: list,
    ):
        """
        Only real arguments are:

        dim: embedding size
        sigma: scale for initializing embedding

        The rest of the arguments are used for constructing the view matrix.

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
        """
        super().__init__()

        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters,
                            offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()  # 3 3

        for i, scale in enumerate(upsample_scales):
            # each decoder block upsamples the bev embedding by a factor of 2
            h = bev_height // scale
            w = bev_width // scale

            # bev coordinates
            grid = generate_grid(h, w).squeeze(0)
            grid[0] = bev_width * grid[0]
            grid[1] = bev_height * grid[1]

            grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')  # 3 (h w)
            grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)  # 3 h w
            # egocentric frame
            self.register_buffer('grid%d'%i, grid, persistent=False)

            # 3 h w
        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim,
                                bev_height//upsample_scales[0],
                                bev_width//upsample_scales[0]))  # d h w

    def get_prior(self):
        return self.learned_features


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 25
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        batch, _, height, width, device, h = *x.shape, x.device, self.heads

        # flatten

        x = rearrange(x, 'b d h w -> b (h w) d')

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h), (q, k, v))

        # scale

        q = q * self.scale

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias

        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b m (h w) d -> b h w (m d)',
                        h = height, w = width)

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, 'b h w d -> b d h w')


class CrossWinAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, rel_pos_emb=False, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head
        self.rel_pos_emb = rel_pos_emb

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)

    def add_rel_pos_emb(self, x):
        return x

    def forward(self, q, k, v, skip=None):
        """
        q: (b n X Y W1 W2 d)
        k: (b n x y w1 w2 d)
        v: (b n x y w1 w2 d)
        return: (b X Y W1 W2 d)
        """
        assert k.shape == v.shape
        _, view_size, q_height, q_width, q_win_height, q_win_width, _ = q.shape
        _, _, kv_height, kv_width, _, _, _ = k.shape
        assert q_height * q_width == kv_height * kv_width

        # flattening
        q = rearrange(q, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        k = rearrange(k, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')
        v = rearrange(v, 'b n x y w1 w2 d -> b (x y) (n w1 w2) d')

        # Project with multiple heads
        q = self.to_q(q)                                # b (X Y) (n W1 W2) (heads dim_head)
        k = self.to_k(k)                                # b (X Y) (n w1 w2) (heads dim_head)
        v = self.to_v(v)                                # b (X Y) (n w1 w2) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b l Q d, b l K d -> b l Q K', q, k)  # b (X Y) (n W1 W2) (n w1 w2)
        # dot = rearrange(dot, 'b l n Q K -> b l Q (n K)')  # b (X Y) (W1 W2) (n w1 w2)

        if self.rel_pos_emb:
            dot = self.add_rel_pos_emb(dot)
        att = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b n Q K, b n K d -> b n Q d', att, v)  # b (X Y) (n W1 W2) d
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)
        a = rearrange(a, ' b (x y) (n w1 w2) d -> b n x y w1 w2 d',
            x=q_height, y=q_width, w1=q_win_height, w2=q_win_width)

        # Combine multiple heads
        z = self.proj(a)

        # reduce n: (b n X Y W1 W2 d) -> (b X Y W1 W2 d)
        z = z.mean(1)  # for sequential usage, we cannot reduce it!

        # Optional skip connection
        if skip is not None:
            z = z + skip
        return z


class CrossViewSwapAttention(nn.Module):
    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        index: int,
        image_height: int,
        image_width: int,
        qkv_bias: bool,
        q_win_size: list,
        feat_win_size: list,
        heads: list,
        dim_head: list,
        bev_embedding_flag: list,
        rel_pos_emb: bool = False,  # to-do
        no_image_features: bool = False,
        skip: bool = True,
        norm=nn.LayerNorm,
    ):
        super().__init__()

        # 1 1 3 h w
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height

        self.register_buffer('image_plane', image_plane, persistent=False)

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, 1, bias=False))

        self.bev_embed_flag = bev_embedding_flag[index]
        if self.bev_embed_flag:
            self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.q_win_size = q_win_size[index]
        self.feat_win_size = feat_win_size[index]
        self.rel_pos_emb = rel_pos_emb

        self.cross_win_attend_1 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.cross_win_attend_2 = CrossWinAttention(dim, heads[index], dim_head[index], qkv_bias)
        self.skip = skip
        # self.proj = nn.Linear(2 * dim, dim)

        self.prenorm_1 = norm(dim)
        self.prenorm_2 = norm(dim)
        self.mlp_1 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.mlp_2 = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def pad_divisble(self, x, win_h, win_w):
        """Pad the x to be divible by window size."""
        _, _, _, h, w = x.shape
        h_pad, w_pad = ((h + win_h) // win_h) * win_h, ((w + win_w) // win_w) * win_w
        padh = h_pad - h if h % win_h != 0 else 0
        padw = w_pad - w if w % win_w != 0 else 0
        return F.pad(x, (0, padw, 0, padh), value=0)

    def forward(
        self,
        index: int,
        x: torch.FloatTensor,
        bev: BEVEmbedding,
        feature: torch.FloatTensor,
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
    ):
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)

        Returns: (b, d, H, W)
        """
        b, n, _, _, _ = feature.shape
        _, _, H, W = x.shape

        pixel = self.image_plane                                                # b n 3 h w
        _, _, _, h, w = pixel.shape

        c = E_inv[..., -1:]                                                     # b n 4 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]                # (b n) 4 1 1
        c_embed = self.cam_embed(c_flat)                                        # (b n) d 1 1

        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                   # 1 1 3 (h w)
        cam = I_inv @ pixel_flat                                                # b n 3 (h w)
        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)                     # b n 4 (h w)
        d = E_inv @ cam                                                         # b n 4 (h w)
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)           # (b n) 4 h w
        d_embed = self.img_embed(d_flat)                                        # (b n) d h w

        img_embed = d_embed - c_embed                                           # (b n) d h w
        img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d h w

        # todo: some hard-code for now.
        if index == 0:
            world = bev.grid0[:2]
        elif index == 1:
            world = bev.grid1[:2]
        elif index == 2:
            world = bev.grid2[:2]
        elif index == 3:
            world = bev.grid3[:2]

        if self.bev_embed_flag:
            # 2 H W
            w_embed = self.bev_embed(world[None])                                   # 1 d H W
            bev_embed = w_embed - c_embed                                           # (b n) d H W
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d H W
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)      # b n d H W

        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')               # (b n) d h w

        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)              # (b n) d h w
        else:
            key_flat = img_embed                                                # (b n) d h w

        val_flat = self.feature_linear(feature_flat)                            # (b n) d h w

        # Expand + refine the BEV embedding
        if self.bev_embed_flag:
            query = query_pos + x[:, None]
        else:
            query = x[:, None]  # b n d H W
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w
        val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)             # b n d h w

        # pad divisible
        key = self.pad_divisble(key, self.feat_win_size[0], self.feat_win_size[1])
        val = self.pad_divisble(val, self.feat_win_size[0], self.feat_win_size[1])

        # local-to-local cross-attention
        query = rearrange(query, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])  # window partition
        key = rearrange(key, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                          w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # window partition
        val = rearrange(val, 'b n d (x w1) (y w2) -> b n x y w1 w2 d',
                          w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # window partition
        query = rearrange(self.cross_win_attend_1(query, key, val,
                                                skip=rearrange(x,
                                                            'b d (x w1) (y w2) -> b x y w1 w2 d',
                                                             w1=self.q_win_size[0], w2=self.q_win_size[1]) if self.skip else None),
                       'b x y w1 w2 d  -> b (x w1) (y w2) d')    # reverse window to feature

        query = query + self.mlp_1(self.prenorm_1(query))

        x_skip = query
        query = repeat(query, 'b x y d -> b n x y d', n=n)              # b n x y d

        # local-to-global cross-attention
        query = rearrange(query, 'b n (x w1) (y w2) d -> b n x y w1 w2 d',
                          w1=self.q_win_size[0], w2=self.q_win_size[1])  # window partition
        key = rearrange(key, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')  # reverse window to feature
        key = rearrange(key, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # grid partition
        val = rearrange(val, 'b n x y w1 w2 d -> b n (x w1) (y w2) d')  # reverse window to feature
        val = rearrange(val, 'b n (w1 x) (w2 y) d -> b n x y w1 w2 d',
                        w1=self.feat_win_size[0], w2=self.feat_win_size[1])  # grid partition
        query = rearrange(self.cross_win_attend_2(query,
                                                  key,
                                                  val,
                                                  skip=rearrange(x_skip,
                                                            'b (x w1) (y w2) d -> b x y w1 w2 d',
                                                            w1=self.q_win_size[0],
                                                            w2=self.q_win_size[1])
                                                  if self.skip else None),
                       'b x y w1 w2 d  -> b (x w1) (y w2) d')  # reverse grid to feature

        query = query + self.mlp_2(self.prenorm_2(query))

        query = self.postnorm(query)

        query = rearrange(query, 'b H W d -> b d H W')

        return query


class FAXModule(nn.Module):
    def __init__(
            self,
            config
    ):
        super().__init__()

        middle = config['middle']
        dim = config['dim']
        self.backbone_output_shape = config['backbone_output_shape']
        assert len(middle) == len(self.backbone_output_shape)

        cross_view = config['cross_view']
        cross_view_swap = config['cross_view_swap']

        cross_views = list()
        layers = list()
        downsample_layers = list()

        for i, (feat_shape, num_layers) in enumerate(zip(self.backbone_output_shape, middle)):
            _, _, _, feat_dim, feat_height, feat_width = \
                torch.zeros(feat_shape).shape

            cva = CrossViewSwapAttention(feat_height, feat_width, feat_dim,
                                         dim[i], i,
                                         **cross_view, **cross_view_swap)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)])
            layers.append(layer)

            if i < len(middle) - 1:
                downsample_layers.append(nn.Sequential(
                    nn.Sequential(
                        nn.Conv2d(dim[i], dim[i] // 4,
                                  kernel_size=3, stride=1,
                                  padding=1, bias=False),
                        nn.PixelUnshuffle(2),
                        nn.Conv2d(dim[i+1], dim[i+1],
                                  3, padding=1, bias=False),
                        nn.BatchNorm2d(dim[i+1]),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(dim[i+1],
                                  dim[i+1], 1, padding=0, bias=False),
                        nn.BatchNorm2d(dim[i+1])
                        )))


        self.bev_embedding = BEVEmbedding(dim[0], **config['bev_embedding'])
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)
        self.downsample_layers = nn.ModuleList(downsample_layers)
        self.self_attn = Attention(dim[-1], **config['self_attn'])

        #CooperTrim channel selection entropy
        self.prev_avg_entropy = None

        #CooperTrim channel select adapt
        num_channel_select = config['channel_select']['channel_dim']
        num_spatial_select = config['channel_select']['spatial_dim']
        self.channel_select_model = CrossAttentionMaskPredictor(num_channels=num_channel_select, spatial_dim=num_spatial_select)
        #CooperTrim curriculum
        self.channel_select_model_adaptive = CrossAttentionMaskPredictorAdaptive(num_channels=num_channel_select, spatial_dim=num_spatial_select)

        #CooperTrim channel select adapt SA
        # self.channel_select_model = SelfAttentionMaskPredictor(num_channels=num_channel_select, spatial_dim=num_spatial_select)

        # CooperTrim learnable confidence level
        initial_confidence_level = 50  # Initial value for the confidence level
        self.confidence_level = nn.Parameter(torch.tensor(initial_confidence_level, dtype=torch.float32))

    #CooperTrim conformal prediction
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

        # CooperTrim learnable confidence level
        quantile_threshold = np.percentile(conformity_scores_np, confidence_level)  # Convert to numpy for quantile calculation
        # quantile_threshold = np.percentile(conformity_scores_np, confidence_level.item())

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
    
    def compute_conformal_uncertainty_adaptive(self, reference_data, current_data, confidence_level=90):
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

        # CooperTrim learnable confidence level
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
    
    # #CooperTrim uncertainty common
    # def compute_uncertainty(reference_data, current_data, task_type, confidence_level=90):
    #     if task_type == "dynamic":
    #         # Conformal prediction for dynamic tasks
    #         conformity_scores = torch.abs(reference_data - current_data).mean(dim=(1, 2))  # Shape: [128]
    #         conformity_scores_np = conformity_scores.detach().cpu().numpy()
    #         quantile_threshold = np.percentile(conformity_scores_np, confidence_level)
    #         uncertainty_flags = conformity_scores > quantile_threshold
    #         uncertainty_intervals = torch.zeros(current_data.shape[0], device=current_data.device)

    #         for i in range(uncertainty_intervals.shape[0]):
    #             if uncertainty_flags[i]:
    #                 lower_bound = conformity_scores[i] - quantile_threshold
    #                 upper_bound = conformity_scores[i] + quantile_threshold
    #                 uncertainty_intervals[i] = upper_bound - lower_bound
    #             else:
    #                 uncertainty_intervals[i] = 0.0
    #         return uncertainty_intervals

    #     elif task_type == "static":
    #         # Standard deviation for static tasks
    #         std_dev = current_data.std(dim=(1, 2))  # Shape: [128]
    #         uncertainty_flags = std_dev > std_dev.mean()  # Example threshold: mean of std_dev
    #         uncertainty_intervals = torch.zeros(current_data.shape[0], device=current_data.device)

    #         for i in range(uncertainty_intervals.shape[0]):
    #             if uncertainty_flags[i]:
    #                 lower_bound = current_data[i].mean().item() - std_dev[i].item()
    #                 upper_bound = current_data[i].mean().item() + std_dev[i].item()
    #                 uncertainty_intervals[i] = upper_bound - lower_bound
    #             else:
    #                 uncertainty_intervals[i] = 0.0
    #         return uncertainty_intervals

    # CooperTrim
    # def forward(self, batch, prev_fused_feature=None): #, ppo_agent=None):

    #CooperTrim epsilon greedy
    def forward(self, batch, epoch, prev_fused_feature=None):
        b, l, n, _, _, _ = batch['inputs'].shape

        I_inv = \
            rearrange(batch['intrinsic'], 'b l m h w -> (b l) m h w').inverse()
        E_inv = rearrange(batch['extrinsic'],
                          'b l m h w -> (b l) m h w')
        features = batch['features']

        x = self.bev_embedding.get_prior()              # d H W
        x = repeat(x, '... -> b ...', b=b * l)  # b*l d H W

        for i, (cross_view, feature, layer) in \
                enumerate(zip(self.cross_views, features, self.layers)):
            feature = rearrange(feature, 'b l n ... -> (b l) n ...', b=b, n=n)

            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv)
            x = layer(x)
            if i < len(features)-1:
                down_sample_block = self.downsample_layers[i]
                x = down_sample_block(x)
        #CooperTrim transform sa fix
        x = self.self_attn(x)

        #CooperTrim select bev points to send to cav
        #assume 30 % data to request
        orig_bev_data_from_all_cav = x
        
       
        data_at_index_0 = x[0]  # Shape: (128, 32, 32)
        dim_len,height, width = data_at_index_0.shape  # Extract H and W

       

        # flattened_data = data_at_index_0.view(dim_len, -1)  # Shape: (128, height * width)
    
        
        num_spatial_indices = data_at_index_0.shape[0]
        select_threshold=None

        # # CooperTrim channel entropy std uncertainty
        # file_path = '/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/channel_usage_cobevt_CA_dyn.txt'

        #CooperTrim epsilon greedy
        epsilon = 0.1  # Exploration probability

        if epoch <=20:
                percentage_data_to_request = 1.0
                num_random_indices = int(percentage_data_to_request * num_spatial_indices)  # Compute 30% of total indices
                #CooperTrim Transmission 1 - this data is transmitted from ego to CAV for request
                # random_indices = torch.randperm(num_spatial_indices, device=flattened_data.device)[:num_random_indices]  # Random 30% indices
                random_indices = torch.arange(num_spatial_indices, device=data_at_index_0.device)[:num_random_indices]
                self.prev_avg_entropy = 1
                percentage_selected = 100.0
                print(f"percentage_selected: {percentage_selected:.2f}%")

                #CooperTrim dump channel select
                # # File path
                # file_path = '/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/channel_usage_cobevt_st_cp50.txt'

                # # Check if the file exists to determine the starting frame
                # if os.path.exists(file_path):
                #     # Read the last line to get the last frame number
                #     with open(file_path, 'r') as file:
                #         lines = file.readlines()
                #         if lines:
                #             last_line = lines[-1]
                #             last_frame = int(last_line.split(',')[0].strip('()'))  # Extract the frame number
                #             current_frame = last_frame + 1
                #         else:
                #             current_frame = 1  # If file is empty, start with frame 1
                # else:
                #     current_frame = 1  # If file doesn't exist, start with frame 1

                # # Prepare the line to be written to the file
                # line_to_write = f"({current_frame},100)\n"

                # # Write to the file
                # with open(file_path, 'a') as file:  # 'a' mode opens the file for appending
                #     file.write(line_to_write)

        elif epoch > 20: # and random.random() > epsilon:  # Exploit with probability
        # if self.prev_avg_entropy is not None:
                
                percentage_data_to_request= 0.05
                # print(f"percentage_data_to_request: {percentage_data_to_request}")

                #CooperTrim prev feature for uncertainty improvement
                # Check if prev_fused_feature is provided
                if prev_fused_feature is not None:
                    uncertainty_intervals = self.compute_conformal_uncertainty(prev_fused_feature, data_at_index_0, confidence_level=50).unsqueeze(0)
                #     # CooperTrim learnable confidence level
                    # uncertainty_intervals = self.compute_conformal_uncertainty(prev_fused_feature, data_at_index_0, confidence_level=self.confidence_level).unsqueeze(0)
                    

                # std_dev = data_at_index_0.std(dim=(1, 2)) 
                # std_dev = std_dev.unsqueeze(0)               

                #CooperTrim channel select adapt
                # sorted_std_dev, sorted_indices = torch.sort(std_dev, descending=True)
                # num_elements = std_dev.shape[0]  # Total number of elements (128 in this case)
                # top_k_percent_count = int(num_elements * percentage_data_to_request)  # Calculate 80% of the elements
                # top_k_indices = sorted_indices[:top_k_percent_count]
                # random_indices = top_k_indices#[torch.randperm(top_k_percent_count)]
                # std_dev = std_dev.unsqueeze(0) 

                # Step 3: Forward Pass

                #CooperTrim conformal prediction
                # predicted_mask, select_threshold = self.channel_select_model(std_dev, data_at_index_0.unsqueeze(0))  # Shape: [batch_size, 128]
                # predicted_mask = self.channel_select_model(std_dev, data_at_index_0.unsqueeze(0)) 
                # print("select_threshold:", select_threshold)
                # predicted_mask, select_threshold = self.channel_select_model(uncertainty_intervals, data_at_index_0.unsqueeze(0)) 
                predicted_mask = self.channel_select_model(uncertainty_intervals, data_at_index_0.unsqueeze(0))
                select_threshold = torch.tensor(0.5)

                
                # Step 4: Convert Probabilities to Binary Mask
                predicted_mask = (predicted_mask > 0.5).float().to(data_at_index_0.device)  # Threshold at 0.5

                indices = torch.where(predicted_mask[0] == 1)#.squeeze(1)
                random_indices = indices[0]#.squeeze(0)
                # print(random_indices.device)
                # random_indices = random_indices.long()

                total_indices = predicted_mask.shape[1]
                selected_indices = random_indices.shape[0]  # Count of selected indices
                percentage_selected = (selected_indices / total_indices) * 100
                print(f"percentage_selected: {percentage_selected:.2f}%")

                # #shilp adump channel select
                # # File path
                # file_path = '/home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/dumps_channel_select/channel_usage_cobevt_st_cp50.txt'

                # # Check if the file exists to determine the starting frame
                # if os.path.exists(file_path):
                #     # Read the last line to get the last frame number
                #     with open(file_path, 'r') as file:
                #         lines = file.readlines()
                #         if lines:
                #             last_line = lines[-1]
                #             last_frame = int(last_line.split(',')[0].strip('()'))  # Extract the frame number
                #             current_frame = last_frame + 1
                #         else:
                #             current_frame = 1  # If file is empty, start with frame 1
                # else:
                #     current_frame = 1  # If file doesn't exist, start with frame 1

                # # Calculate the percentage of selected indices
                # total_indices = random_indices.numel()
                # selected_indices = random_indices.sum().item()  # Count of selected indices
                # percentage_selected = (selected_indices / total_indices) * 100

                # # Prepare the line to be written to the file
                # line_to_write = f"({current_frame},{percentage_selected})\n"

                # # Write to the file
                # with open(file_path, 'a') as file:  # 'a' mode opens the file for appending
                #     file.write(line_to_write)
                # #    print(f"Frame {current_frame}: {percentage_selected:.2f}% of indices selected")          
        elif epoch > 60: # and random.random() > epsilon:  # Exploit with probability
        # if self.prev_avg_entropy is not None:
                
                percentage_data_to_request= 0.05
                # print(f"percentage_data_to_request: {percentage_data_to_request}")

                #CooperTrim prev feature for uncertainty improvement
                # Check if prev_fused_feature is provided
                if prev_fused_feature is not None:
                    uncertainty_intervals = self.compute_conformal_uncertainty(prev_fused_feature, data_at_index_0, confidence_level=50).unsqueeze(0)
                #     # CooperTrim learnable confidence level
                    # uncertainty_intervals = self.compute_conformal_uncertainty(prev_fused_feature, data_at_index_0, confidence_level=self.confidence_level).unsqueeze(0)
                    

                # std_dev = data_at_index_0.std(dim=(1, 2)) 
                # std_dev = std_dev.unsqueeze(0)               

                #CooperTrim channel select adapt
                # sorted_std_dev, sorted_indices = torch.sort(std_dev, descending=True)
                # num_elements = std_dev.shape[0]  # Total number of elements (128 in this case)
                # top_k_percent_count = int(num_elements * percentage_data_to_request)  # Calculate 80% of the elements
                # top_k_indices = sorted_indices[:top_k_percent_count]
                # random_indices = top_k_indices#[torch.randperm(top_k_percent_count)]
                # std_dev = std_dev.unsqueeze(0) 

                # Step 3: Forward Pass

                #CooperTrim conformal prediction
                # predicted_mask, select_threshold = self.channel_select_model(std_dev, data_at_index_0.unsqueeze(0))  # Shape: [batch_size, 128]
                # predicted_mask = self.channel_select_model(std_dev, data_at_index_0.unsqueeze(0)) 
                # print("select_threshold:", select_threshold)
                predicted_mask, select_threshold = self.channel_select_model_adaptive(uncertainty_intervals, data_at_index_0.unsqueeze(0)) 
                # predicted_mask = self.channel_select_model(uncertainty_intervals, data_at_index_0.unsqueeze(0))
                # select_threshold = torch.tensor(0.5)

                
                # Step 4: Convert Probabilities to Binary Mask
                # predicted_mask = (predicted_mask > 0.5).float().to(data_at_index_0.device)  # Threshold at 0.5

                indices = torch.where(predicted_mask[0] == 1)#.squeeze(1)
                random_indices = indices[0]#.squeeze(0)
                # print(random_indices.device)
                # random_indices = random_indices.long()

                total_indices = predicted_mask.shape[1]
                selected_indices = random_indices.shape[0]  # Count of selected indices
                percentage_selected = (selected_indices / total_indices) * 100
                print(f"percentage_selected: {percentage_selected:.2f}%")

        elif epoch>=100: # and random.random() > epsilon:  # Exploit with probability
        # if self.prev_avg_entropy is not None:
                
                percentage_data_to_request= 0.05
                # print(f"percentage_data_to_request: {percentage_data_to_request}")

                #CooperTrim prev feature for uncertainty improvement
                # Check if prev_fused_feature is provided
                if prev_fused_feature is not None:
                    # uncertainty_intervals = self.compute_conformal_uncertainty(prev_fused_feature, data_at_index_0, confidence_level=50).unsqueeze(0)
                #     # CooperTrim learnable confidence level
                    uncertainty_intervals = self.compute_conformal_uncertainty_adaptive(prev_fused_feature, data_at_index_0, confidence_level=self.confidence_level).unsqueeze(0)
                    

                # std_dev = data_at_index_0.std(dim=(1, 2)) 
                # std_dev = std_dev.unsqueeze(0)               

                #CooperTrim channel select adapt
                # sorted_std_dev, sorted_indices = torch.sort(std_dev, descending=True)
                # num_elements = std_dev.shape[0]  # Total number of elements (128 in this case)
                # top_k_percent_count = int(num_elements * percentage_data_to_request)  # Calculate 80% of the elements
                # top_k_indices = sorted_indices[:top_k_percent_count]
                # random_indices = top_k_indices#[torch.randperm(top_k_percent_count)]
                # std_dev = std_dev.unsqueeze(0) 

                # Step 3: Forward Pass

                #CooperTrim conformal prediction
                # predicted_mask, select_threshold = self.channel_select_model(std_dev, data_at_index_0.unsqueeze(0))  # Shape: [batch_size, 128]
                # predicted_mask = self.channel_select_model(std_dev, data_at_index_0.unsqueeze(0)) 
                # print("select_threshold:", select_threshold)
                predicted_mask, select_threshold = self.channel_select_model_adaptive(uncertainty_intervals, data_at_index_0.unsqueeze(0)) 
                # predicted_mask = self.channel_select_model(uncertainty_intervals, data_at_index_0.unsqueeze(0))
                # select_threshold = torch.tensor(0.5)

                
                # Step 4: Convert Probabilities to Binary Mask
                # predicted_mask = (predicted_mask > 0.5).float().to(data_at_index_0.device)  # Threshold at 0.5

                indices = torch.where(predicted_mask[0] == 1)#.squeeze(1)
                random_indices = indices[0]#.squeeze(0)
                # print(random_indices.device)
                # random_indices = random_indices.long()

                total_indices = predicted_mask.shape[1]
                selected_indices = random_indices.shape[0]  # Count of selected indices
                percentage_selected = (selected_indices / total_indices) * 100
                print(f"percentage_selected: {percentage_selected:.2f}%")
                
                            

        # percentage_data_to_request = 0.05
        # # print(f"percentage_data_to_request: {percentage_data_to_request}")
        # num_random_indices = int(percentage_data_to_request * num_spatial_indices)  # Compute 30% of total indices
        # #CooperTrim Transmission 1 - this data is transmitted from ego to CAV for request
        # random_indices = torch.randperm(num_spatial_indices, device=flattened_data.device)[:num_random_indices]  # Random 30% indices
        # # random_indices = torch.arange(num_spatial_indices, device=flattened_data.device)[:num_random_indices]
        # self.prev_avg_entropy = 1
        # normalized_uncertainty = 1.0     
        
        
        #CooperTrim channel entropy soft
        return orig_bev_data_from_all_cav, random_indices, select_threshold, percentage_selected
        # return orig_bev_data_from_all_cav, random_indices
        # return orig_bev_data_from_all_cav, random_indices, channel_select_probabilities, percentage_selected, std_dev 

    #CooperTrim grad cam
    # Add this to your code after the forward function
    def visualize_selected_channels(self, orig_bev_data_from_all_cav, random_indices, output_dir, frame_counter=None):
        """
        Visualize the selected channels from the BEV feature tensor
        
        Args:
            orig_bev_data_from_all_cav: Feature tensor from the model
            random_indices: Selected channel indices
            output_dir: Directory to save visualizations
            frame_counter: Optional frame counter for tracking (default: None)
        """
        # Create a unique frame ID if not provided
        if frame_counter is None:
            # Use timestamp as a unique identifier
            import time
            frame_id = int(time.time())
        else:
            frame_id = frame_counter
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Process and save visualizations
        process_feature_visualization(
            orig_bev_data_from_all_cav, 
            random_indices,
            output_dir,
            frame_id
        )
        
        return frame_id

