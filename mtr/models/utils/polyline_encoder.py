# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import torch
import torch.nn as nn
from ..utils import common_layers


class PointNetPolylineEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=1, out_channels=None):
        super().__init__()
        self.pre_mlps = common_layers.build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=False
        )
        self.mlps = common_layers.build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim] * (num_layers - num_pre_layers),
            ret_before_act=False
        )
        
        if out_channels is not None:
            self.out_mlps = common_layers.build_mlps(
                c_in=hidden_dim, mlp_channels=[hidden_dim, out_channels], 
                ret_before_act=True, without_norm=True
            )
        else:
            self.out_mlps = None 

    def forward(self, polylines, polylines_mask):
        """
        Args:
            polylines (batch_size, num_polylines, num_points_each_polylines, C):
            polylines_mask (batch_size, num_polylines, num_points_each_polylines):

        Returns:
        """
        batch_size, num_polylines,  num_points_each_polylines, C = polylines.shape

        # pre-mlp
        polylines_feature_valid = self.pre_mlps(polylines[polylines_mask])  # (N, C)
        polylines_feature = polylines.new_zeros(batch_size, num_polylines,  num_points_each_polylines, polylines_feature_valid.shape[-1])
        polylines_feature[polylines_mask] = polylines_feature_valid

        # get global feature
        pooled_feature = polylines_feature.max(dim=2)[0]
        polylines_feature = torch.cat((polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, num_points_each_polylines, 1)), dim=-1)

        # mlp
        polylines_feature_valid = self.mlps(polylines_feature[polylines_mask])
        feature_buffers = polylines_feature.new_zeros(batch_size, num_polylines, num_points_each_polylines, polylines_feature_valid.shape[-1])
        feature_buffers[polylines_mask] = polylines_feature_valid

        # max-pooling
        feature_buffers = feature_buffers.max(dim=2)[0]  # (batch_size, num_polylines, C)
        
        # out-mlp 
        if self.out_mlps is not None:
            valid_mask = (polylines_mask.sum(dim=-1) > 0)
            feature_buffers_valid = self.out_mlps(feature_buffers[valid_mask])  # (N, C)
            feature_buffers = feature_buffers.new_zeros(batch_size, num_polylines, feature_buffers_valid.shape[-1])
            feature_buffers[valid_mask] = feature_buffers_valid
        return feature_buffers


class FourierPositionEncoder(nn.Module):
    """Polar + Fourier position embedding (QCNet-style).

    Converts (x, y) positions to polar (r, theta) then applies a
    Fourier feature expansion to improve spatial generalisation.
    """

    def __init__(self, out_dim, num_freqs=8, scale=10.0):
        super().__init__()
        self.num_freqs = num_freqs
        self.scale = scale
        freqs = 2.0 ** torch.arange(num_freqs).float() * scale  # (num_freqs,)
        self.register_buffer('freqs', freqs)
        self.proj = nn.Linear(4 * num_freqs, out_dim)

    def forward(self, pos):
        """pos: (..., 2) cartesian -> (..., out_dim) Fourier features."""
        r = pos.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        theta = torch.atan2(pos[..., 1:2], pos[..., 0:1])
        polar = torch.cat([r, theta], dim=-1)  # (..., 2)

        feats = []
        for i in range(self.num_freqs):
            f = self.freqs[i]
            feats.append(torch.sin(polar * f))
            feats.append(torch.cos(polar * f))
        feats = torch.cat(feats, dim=-1)  # (..., 4*num_freqs)
        return self.proj(feats)


class PointNetPolylineEncoderWithFourier(nn.Module):
    """Polyline encoder with optional polar+Fourier position injection.

    Behaves identically to ``PointNetPolylineEncoder`` when
    ``use_fourier_pos=False``. When enabled, it concatenates Fourier
    position features to the per-point features before the pre-MLP.
    """

    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=1,
                 out_channels=None, use_fourier_pos=False, fourier_freqs=8):
        super().__init__()
        self.use_fourier_pos = use_fourier_pos
        if use_fourier_pos:
            self.fourier_encoder = FourierPositionEncoder(out_dim=hidden_dim, num_freqs=fourier_freqs)
            in_channels = in_channels + hidden_dim

        self.pre_mlps = common_layers.build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=False
        )
        self.mlps = common_layers.build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim] * (num_layers - num_pre_layers),
            ret_before_act=False
        )

        if out_channels is not None:
            self.out_mlps = common_layers.build_mlps(
                c_in=hidden_dim, mlp_channels=[hidden_dim, out_channels],
                ret_before_act=True, without_norm=True
            )
        else:
            self.out_mlps = None

    def forward(self, polylines, polylines_mask):
        batch_size, num_polylines, num_points_each_polylines, C = polylines.shape

        if self.use_fourier_pos:
            pos_feat = self.fourier_encoder(polylines[..., 0:2])  # (B, P, N, hidden)
            polylines = torch.cat((polylines, pos_feat), dim=-1)

        # pre-mlp
        polylines_feature_valid = self.pre_mlps(polylines[polylines_mask])
        polylines_feature = polylines.new_zeros(batch_size, num_polylines, num_points_each_polylines, polylines_feature_valid.shape[-1])
        polylines_feature[polylines_mask] = polylines_feature_valid

        # get global feature
        pooled_feature = polylines_feature.max(dim=2)[0]
        polylines_feature = torch.cat((polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, num_points_each_polylines, 1)), dim=-1)

        # mlp
        polylines_feature_valid = self.mlps(polylines_feature[polylines_mask])
        feature_buffers = polylines_feature.new_zeros(batch_size, num_polylines, num_points_each_polylines, polylines_feature_valid.shape[-1])
        feature_buffers[polylines_mask] = polylines_feature_valid

        # max-pooling
        feature_buffers = feature_buffers.max(dim=2)[0]

        # out-mlp
        if self.out_mlps is not None:
            valid_mask = (polylines_mask.sum(dim=-1) > 0)
            feature_buffers_valid = self.out_mlps(feature_buffers[valid_mask])
            feature_buffers = feature_buffers.new_zeros(batch_size, num_polylines, feature_buffers_valid.shape[-1])
            feature_buffers[valid_mask] = feature_buffers_valid
        return feature_buffers
