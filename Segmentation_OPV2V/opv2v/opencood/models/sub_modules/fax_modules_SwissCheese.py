import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck
from typing import List

#shilpa entropy
import numpy as np
#shilpa channel select adapt
from opencood.models.sub_modules.channel_select_attention import CrossAttentionMaskPredictor, CrossAttentionMaskPredictorAdaptive
# from opencood.models.sub_modules.channel_select_attention_ppo import CrossAttentionMaskPredictor
import os

#shilpa channel select adapt SA
# from opencood.models.sub_modules.channel_select_self_attention import SelfAttentionMaskPredictor
# import os

#shilpa grad cam
from opencood.models.sub_modules.inference_grad_cam import process_feature_visualization

#shilpa epsilon greedy
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


class FAXModule_SwissCheese(nn.Module):
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

        # Fine-Grained Collaborative Attention (FGCA) Components
        channel_dim = dim[-1]
        fgca_config = config.get('fgca', {}) # Use .get() to avoid KeyError if not present
        channel_reduction_ratio = fgca_config.get('channel_reduction_ratio', 4)
        spatial_kernel_size = fgca_config.get('spatial_kernel_size', 7)
        spatial_padding = fgca_config.get('spatial_padding', 3)

        self.mlp_channel_avg = nn.Sequential(
            nn.Conv2d(channel_dim, channel_dim // channel_reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_dim // channel_reduction_ratio, channel_dim, kernel_size=1, bias=False)
        )
        self.mlp_channel_max = nn.Sequential(
            nn.Conv2d(channel_dim, channel_dim // channel_reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_dim // channel_reduction_ratio, channel_dim, kernel_size=1, bias=False)
        )
        self.mlp_channel_uni = nn.Sequential(
            nn.Conv2d(channel_dim, channel_dim // channel_reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_dim // channel_reduction_ratio, channel_dim, kernel_size=1, bias=False)
        )

        # Spatial Attention Conv
        self.conv_spatial = nn.Conv2d(3, 1, kernel_size=spatial_kernel_size, padding=spatial_padding, bias=False)

        # Dual-Dimensional Feature Selection (DDFS) Hyperparameters
        ddfs_config = config.get('ddfs', {})
        self.alpha = nn.Parameter(torch.tensor(ddfs_config.get('alpha', 1.0), dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(ddfs_config.get('beta', 1.0), dtype=torch.float32))
        self.K0 = nn.Parameter(torch.tensor(ddfs_config.get('K0', 1.0), dtype=torch.float32))
        self.filter_ratio = ddfs_config.get('filter_ratio', 0.25)  # Default to 25% selection if not specified
        self.IL = ddfs_config.get('IL', 1.0)  # Default value for IL
        self.p = ddfs_config.get('p', 0.5)    # Default value for p

    def compute_kurtosis(self, x):
        """Compute kurtosis of a tensor along spatial dimensions."""
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        var = torch.mean((x - mean) ** 2, dim=(2, 3), keepdim=True)
        kurt = torch.mean(((x - mean) ** 4) / (var ** 2 + 1e-8), dim=(2, 3), keepdim=True) - 3
        return kurt

    def compute_skewness(self, x):
        """Compute skewness of a tensor along spatial dimensions."""
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        var = torch.mean((x - mean) ** 2, dim=(2, 3), keepdim=True)
        skew = torch.mean(((x - mean) ** 3) / (var ** 1.5 + 1e-8), dim=(2, 3), keepdim=True)
        return skew

    def adaptive_adjust(self, Pi, Ci):
        """
        Adaptively adjust the filtering ratio Pi based on statistical properties of Ci.
        Implements the formula from OCR: P_i = P_i * W_stat * W_comm
        """
        # Compute statistical properties of Ci
        kurt_ci = self.compute_kurtosis(Ci)
        skew_ci = self.compute_skewness(Ci)

        # Compute W_stat
        stat_term = (kurt_ci / self.alpha + skew_ci / self.beta)
        W_stat = 1 + stat_term * (self.K0 - self.p) / self.IL

        # Compute W_comm
        W_comm = 1 + (1 - Pi) / 2

        # Adjust Pi to P_prime
        P_prime = Pi * W_stat.mean() * W_comm  # Use mean to reduce to scalar if necessary

        # Clamp P_prime to ensure it stays within valid range [0, 1]
        P_prime = torch.clamp(P_prime, min=0.0, max=1.0)

        return P_prime.item() if isinstance(P_prime, torch.Tensor) else P_prime

    def uniq_indicator(self, Mfg_i, Mi, Pi_prime):
        """
        Compute the uniqueness indicator Oi for channel selection at each spatial location.
        Based on the formula: Oi[h, w] = Vi[h, w] * (Pi_prime * H * W * C) / Sum(Vi)
        where Vi = Var(Mfg) * Mi
        """
        # Shape of Mfg_i: (1, C, H, W)
        # Shape of Mi: (1, 1, H, W)
        H, W = Mfg_i.shape[2], Mfg_i.shape[3]
        C = Mfg_i.shape[1]

        # Compute variance across channel dimension for each spatial location
        Vi = torch.var(Mfg_i, dim=1, keepdim=True)  # Shape: (1, 1, H, W)
        Vi = Vi * Mi  # Modulate by spatial mask Mi, Shape: (1, 1, H, W)

        # Compute total sum of Vi across spatial dimensions
        sum_Vi = torch.sum(Vi)  # Scalar value

        # Compute scaling factor with Sum(Vi) in the denominator, using Pi_prime
        if sum_Vi > 0:  # Avoid division by zero
            scaling_factor = (Pi_prime * H * W * C) / sum_Vi
        else:
            scaling_factor = 0.0  # If no variance, no channels are selected

        # Compute Oi[h, w] for each spatial location
        Oi = Vi * scaling_factor  # Shape: (1, 1, H, W)
        # Clamp Oi to ensure it doesn't exceed total channels C at each location
        Oi = torch.clamp(Oi, min=0, max=C)

        return Oi

    def adaptive_adjust(self, Pi_prime, Ci):
        """
        Adaptively adjust the filtering proportion based on statistical properties of Ci.
        This adjusts Pi_prime to P_prime based on the distribution of attention scores in Ci.
        """
        # Compute statistical measures of Ci for adjustment
        mean_Ci = torch.mean(Ci)
        std_Ci = torch.std(Ci)
        # Adjust Pi_prime based on the distribution (example logic)
        # If Ci has high variance, retain more spatial locations (increase P_prime)
        # If Ci is uniform, retain fewer (decrease P_prime)
        adjustment_factor = 1.0 + (std_Ci / (mean_Ci + 1e-6)) * 0.1  # Example adjustment
        P_prime = Pi_prime * adjustment_factor
        # Clamp to ensure P_prime remains within valid range [0, 1]
        P_prime = torch.clamp(P_prime, min=0.0, max=1.0)
        return P_prime


    def forward(self, batch):
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
        # shilpa transform sa fix
        x = self.self_attn(x)

        # shilpa select bev points to send to cav
        orig_bev_data_from_all_cav = x

        # Implementing Fine-Grained Collaborative Attention Algorithm
        # Input: Feature map F (orig_bev_data_from_all_cav)
        F = orig_bev_data_from_all_cav  # Shape: (b*l, C, H, W)

        # Channel Attention Module
        F_avg = torch.mean(F, dim=(2, 3), keepdim=True)  # Avg Pool over spatial dimensions
        F_max = torch.max(F, dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]  # Max Pool over spatial dimensions
        F_uni = torch.mean(F, dim=(2, 3), keepdim=True)  # Assuming Uni Pool is similar to Avg Pool for now (as per OCR)

        # MLP operations (simulated with linear layers or conv1x1 for simplicity)
        # In practice, you might need to define these MLPs as part of your model architecture
        Mc_avg = self.mlp_channel_avg(F_avg) if hasattr(self, 'mlp_channel_avg') else F_avg
        Mc_max = self.mlp_channel_max(F_max) if hasattr(self, 'mlp_channel_max') else F_max
        Mc_uni = self.mlp_channel_uni(F_uni) if hasattr(self, 'mlp_channel_uni') else F_uni

        Mc = Mc_avg + Mc_max + Mc_uni  # Combine the channel attention features
        Mc = torch.sigmoid(Mc)  # Sigmoid activation to get channel attention map
        F_prime = Mc * F  # Apply channel attention to original feature map

        # Spatial Attention Module
        F_avg_spatial = torch.mean(F_prime, dim=1, keepdim=True)  # Avg Pool over channel dimension
        F_max_spatial = torch.max(F_prime, dim=1, keepdim=True)[0]  # Max Pool over channel dimension
        F_uni_spatial = torch.mean(F_prime, dim=1, keepdim=True)  # Assuming Uni Pool similar to Avg Pool

        F_concat = torch.cat([F_avg_spatial, F_max_spatial, F_uni_spatial], dim=1)  # Concatenate along channel dimension
        Ms = self.conv_spatial(F_concat) if hasattr(self, 'conv_spatial') else F_concat  # Conv operation (define in model)
        Ms = torch.sigmoid(Ms)  # Sigmoid activation to get spatial attention map

        # Fine-grained collaborative attention map
        Mfg = Mc * Ms  # Element-wise multiplication of channel and spatial attention maps

        # Dual-Dimensional Feature Selection (DDFS) for each vehicle
        selected_features_list = []
        total_vehicles = b * l  # Assuming each (b*l) corresponds to a vehicle or a batch element
        bandwidths = self.get_bandwidths(total_vehicles) if hasattr(self, 'get_bandwidths') else [40] * total_vehicles  # Placeholder for bandwidths

        for vehicle_i in range(total_vehicles):
            Fi = F[vehicle_i:vehicle_i+1]  # Feature map for vehicle i, shape: (1, C, H, W)
            Mfg_i = Mfg[vehicle_i:vehicle_i+1]  # FGCA map for vehicle i
            Bi = bandwidths[vehicle_i]  # Available bandwidth for vehicle i

            # Initialize selection mask Si with the same dimensions as Fi
            Si = torch.zeros_like(Fi)  # Shape: (1, C, H, W)

            # Step 1: Channel Selection Preparation
            Ci = torch.max(Mfg_i, dim=1, keepdim=True)[0]  # Max-pooling along channel dimension
            Ci = torch.nn.functional.normalize(Ci, dim=(2, 3))  # Normalize the spatial attention map

            # Step 2: Determine Filtering Ratio based on Bandwidth
            # Pi = self.filter_ratio(Bi) if hasattr(self, 'filter_ratio') else Bi  # Placeholder for filter ratio function
            Pi_prime = self.filter_ratio if hasattr(self, 'filter_ratio') else 1.0

            if Pi_prime == 0:
                Si = torch.zeros_like(Mfg_i)  # No features selected
                selected_features = Si * Fi
                selected_features_list.append(selected_features)
                continue
            elif Pi_prime == 1:
                Si = Mfg_i  # All features selected
                selected_features = Si * Fi
                selected_features_list.append(selected_features)
                continue

            # Adaptive Adjustment of proportion based on Ci statistics
            P_prime = self.adaptive_adjust(Pi_prime, Ci)  # Compute adjusted proportion P_prime

            # Compute threshold tau based on P_prime
            H, W = Ci.shape[2], Ci.shape[3]
            Csort = torch.sort(Ci.view(-1), descending=True)[0]  # Sort Ci in descending order
            target_count = int(P_prime * H * W)  # Number of spatial locations to retain based on P_prime
            tau = Csort[target_count - 1] if target_count > 0 else Csort[-1]  # Threshold based on P_prime


            # Step 5: Create Spatial Selection Mask Mi
            Mi = (Ci > tau).float()  # Binary mask where values > tau are 1, others 0

            # Step 6: Channel Selection - Compute uniqueness indicator Oi
            Oi = self.uniq_indicator(Mfg_i, Mi, Pi_prime)  # Use original Pi_prime for channel selection scaling

            # Step 7: Apply Channel Selection based on Bandwidth Constraints
            for h in range(H):
                for w in range(W):
                    if Oi[0, 0, h, w] > 0:  # Non-zero spatial location
                        num_channels_to_select = int(Oi[0, 0, h, w].item())
                        if num_channels_to_select >= Mfg_i.shape[1]:  # Sufficient bandwidth
                            Si[0, :, h, w] = Mfg_i[0, :, h, w]
                        else:  # Insufficient bandwidth, select top-k channels
                            values, indices = torch.topk(Mfg_i[0, :, h, w], k=num_channels_to_select)
                            Si[0, :, h, w] = 0
                            Si[0, indices, h, w] = Mfg_i[0, indices, h, w]

            # Step 8: Final Selected Features for vehicle i
            F_selected_i = Si * Fi
            selected_features_list.append(F_selected_i)

        # Stack selected features for all vehicles
        enhanced_features = torch.cat(selected_features_list, dim=0)

        # Placeholder outputs to match original function signature
        percentage_selected = 100.0 * Pi_prime    # Approximate based on last vehicle's Pi
        # random_indices = torch.arange(F.shape[1], device=F.device)  # Placeholder
        select_threshold = tau if 'tau' in locals() else None  # Placeholder

        print(f"percentage_selected: {percentage_selected:.2f}%")

        return enhanced_features, select_threshold, percentage_selected



