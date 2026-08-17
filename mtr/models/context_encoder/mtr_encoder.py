# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import numpy as np
import torch
import torch.nn as nn


from mtr.models.utils.transformer import transformer_encoder_layer, position_encoding_utils
from mtr.models.utils import polyline_encoder
from mtr.utils import common_utils
from mtr.ops.knn import knn_utils


class MTREncoder(nn.Module):
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__()
        self.model_cfg = config

        # build polyline encoders
        self.agent_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_AGENT + 1,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_AGENT,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_AGENT,
            out_channels=self.model_cfg.D_MODEL
        )
        self.map_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_MAP,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_MAP,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_MAP,
            num_pre_layers=self.model_cfg.NUM_LAYER_IN_PRE_MLP_MAP,
            out_channels=self.model_cfg.D_MODEL
        )

        # build transformer encoder layers
        self.use_local_attn = self.model_cfg.get('USE_LOCAL_ATTN', False)
        self_attn_layers = []
        for _ in range(self.model_cfg.NUM_ATTN_LAYERS):
            self_attn_layers.append(self.build_transformer_encoder_layer(
                d_model=self.model_cfg.D_MODEL,
                nhead=self.model_cfg.NUM_ATTN_HEAD,
                dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
                normalize_before=False,
                use_local_attn=self.use_local_attn
            ))

        self.self_attn_layers = nn.ModuleList(self_attn_layers)
        self.num_out_channels = self.model_cfg.D_MODEL

    def build_polyline_encoder(self, in_channels, hidden_dim, num_layers, num_pre_layers=1, out_channels=None):
        use_fourier = self.model_cfg.get('USE_FOURIER_POS', False)
        fourier_freqs = self.model_cfg.get('FOURIER_NUM_FREQS', 8)
        if use_fourier:
            ret_polyline_encoder = polyline_encoder.PointNetPolylineEncoderWithFourier(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_pre_layers=num_pre_layers,
                out_channels=out_channels,
                use_fourier_pos=True,
                fourier_freqs=fourier_freqs,
            )
        else:
            ret_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_pre_layers=num_pre_layers,
                out_channels=out_channels
            )
        return ret_polyline_encoder

    def build_transformer_encoder_layer(self, d_model, nhead, dropout=0.1, normalize_before=False, use_local_attn=False):
        single_encoder_layer = transformer_encoder_layer.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout,
            normalize_before=normalize_before, use_local_attn=use_local_attn
        )
        return single_encoder_layer

    def apply_global_attn(self, x, x_mask, x_pos):
        """

        Args:
            x (batch_size, N, d_model):
            x_mask (batch_size, N):
            x_pos (batch_size, N, 3):
        """
        assert torch.all(x_mask.sum(dim=-1) > 0)

        batch_size, N, d_model = x.shape
        x_t = x.permute(1, 0, 2)
        x_mask_t = x_mask.permute(1, 0, 2)
        x_pos_t = x_pos.permute(1, 0, 2)
 
        pos_embedding = position_encoding_utils.gen_sineembed_for_position(x_pos_t, hidden_dim=d_model)

        for k in range(len(self.self_attn_layers)):
            x_t = self.self_attn_layers[k](
                src=x_t,
                src_key_padding_mask=~x_mask_t,
                pos=pos_embedding
            )
        x_out = x_t.permute(1, 0, 2)  # (batch_size, N, d_model)
        return x_out

    def apply_local_attn(self, x, x_mask, x_pos, num_of_neighbors):
        """

        Args:
            x (batch_size, N, d_model):
            x_mask (batch_size, N):
            x_pos (batch_size, N, 3):
        """
        assert torch.all(x_mask.sum(dim=-1) > 0)
        batch_size, N, d_model = x.shape

        x_stack_full = x.view(-1, d_model)  # (batch_size * N, d_model)
        x_mask_stack = x_mask.view(-1)
        x_pos_stack_full = x_pos.view(-1, 3)
        batch_idxs_full = torch.arange(batch_size).type_as(x)[:, None].repeat(1, N).view(-1).int()  # (batch_size * N)

        # filter invalid elements
        x_stack = x_stack_full[x_mask_stack]
        x_pos_stack = x_pos_stack_full[x_mask_stack]
        batch_idxs = batch_idxs_full[x_mask_stack]

        # knn
        batch_offsets = common_utils.get_batch_offsets(batch_idxs=batch_idxs, bs=batch_size).int()  # (batch_size + 1)
        batch_cnt = batch_offsets[1:] - batch_offsets[:-1]

        index_pair = knn_utils.knn_batch_mlogk(
            x_pos_stack, x_pos_stack,  batch_idxs, batch_offsets, num_of_neighbors
        )  # (num_valid_elems, K)

        # positional encoding
        pos_embedding = position_encoding_utils.gen_sineembed_for_position(x_pos_stack[None, :, 0:2], hidden_dim=d_model)[0]

        # local attn
        output = x_stack
        for k in range(len(self.self_attn_layers)):
            output = self.self_attn_layers[k](
                src=output,
                pos=pos_embedding,
                index_pair=index_pair,
                query_batch_cnt=batch_cnt,
                key_batch_cnt=batch_cnt,
                index_pair_batch=batch_idxs
            )

        ret_full_feature = torch.zeros_like(x_stack_full)  # (batch_size * N, d_model)
        ret_full_feature[x_mask_stack] = output

        ret_full_feature = ret_full_feature.view(batch_size, N, d_model)
        return ret_full_feature

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
              input_dict:
        """
        input_dict = batch_dict['input_dict']
        obj_trajs, obj_trajs_mask = input_dict['obj_trajs'].to(self.device), input_dict['obj_trajs_mask'].to(self.device) 
        map_polylines, map_polylines_mask = input_dict['map_polylines'].to(self.device), input_dict['map_polylines_mask'].to(self.device) 

        obj_trajs_last_pos = input_dict['obj_trajs_last_pos'].to(self.device) 
        map_polylines_center = input_dict['map_polylines_center'].to(self.device) 
        track_index_to_predict = input_dict['track_index_to_predict']

        assert obj_trajs_mask.dtype == torch.bool and map_polylines_mask.dtype == torch.bool

        num_center_objects, num_objects, num_timestamps, _ = obj_trajs.shape
        num_polylines = map_polylines.shape[1]

        # apply polyline encoder
        obj_trajs_in = torch.cat((obj_trajs, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1)
        obj_polylines_feature = self.agent_polyline_encoder(obj_trajs_in, obj_trajs_mask)  # (num_center_objects, num_objects, C)
        map_polylines_feature = self.map_polyline_encoder(map_polylines, map_polylines_mask)  # (num_center_objects, num_polylines, C)

        # apply self-attn
        obj_valid_mask = (obj_trajs_mask.sum(dim=-1) > 0)  # (num_center_objects, num_objects)
        map_valid_mask = (map_polylines_mask.sum(dim=-1) > 0)  # (num_center_objects, num_polylines)

        global_token_feature = torch.cat((obj_polylines_feature, map_polylines_feature), dim=1) 
        global_token_mask = torch.cat((obj_valid_mask, map_valid_mask), dim=1) 
        global_token_pos = torch.cat((obj_trajs_last_pos, map_polylines_center), dim=1) 

        if self.use_local_attn:
            global_token_feature = self.apply_local_attn(
                x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos,
                num_of_neighbors=self.model_cfg.NUM_OF_ATTN_NEIGHBORS
            )
        else:
            global_token_feature = self.apply_global_attn(
                x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos
            )

        obj_polylines_feature = global_token_feature[:, :num_objects]
        map_polylines_feature = global_token_feature[:, num_objects:]
        assert map_polylines_feature.shape[1] == num_polylines

        # organize return features
        center_objects_feature = obj_polylines_feature[torch.arange(num_center_objects), track_index_to_predict]

        batch_dict['center_objects_feature'] = center_objects_feature
        batch_dict['obj_feature'] = obj_polylines_feature
        batch_dict['map_feature'] = map_polylines_feature
        batch_dict['obj_mask'] = obj_valid_mask
        batch_dict['map_mask'] = map_valid_mask
        batch_dict['obj_pos'] = obj_trajs_last_pos
        batch_dict['map_pos'] = map_polylines_center

        return batch_dict


class SharedSceneEncoder(nn.Module):
    """QCNet-style scene-level shared encoder with factorized local attention.

    Uses the same KNN-based local attention as ``MTREncoder`` (not global
    attention) to keep memory usage O(N·K) instead of O(N²).  The factorized
    structure applies three decoupled passes:

    1. **Agent self-attention** — agents attend to nearby agents.
    2. **Map self-attention** — map polylines attend to nearby polylines.
    3. **Agent-map cross-attention** — agents attend to nearby map polylines.

    All passes are batched across center objects (no per-object loop) and use
    local attention for memory efficiency.  Polar + Fourier position encoding
    is optionally injected into polyline features.
    """

    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__()
        self.model_cfg = config

        d_model = self.model_cfg.D_MODEL
        nhead = self.model_cfg.NUM_ATTN_HEAD
        dropout = self.model_cfg.get('DROPOUT_OF_ATTN', 0.1)
        num_attn_layers = self.model_cfg.NUM_ATTN_LAYERS

        # build polyline encoders
        self.agent_polyline_encoder = self._build_polyline(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_AGENT + 1,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_AGENT,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_AGENT,
            out_channels=d_model
        )
        self.map_polyline_encoder = self._build_polyline(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_MAP,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_MAP,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_MAP,
            num_pre_layers=self.model_cfg.NUM_LAYER_IN_PRE_MLP_MAP,
            out_channels=d_model
        )

        # factorized local attention layers (reuse MTREncoder's TransformerEncoderLayer)
        self.agent_attn_layers = nn.ModuleList([
            self._build_encoder_layer(d_model, nhead, dropout, use_local_attn=True)
            for _ in range(num_attn_layers)
        ])
        self.map_attn_layers = nn.ModuleList([
            self._build_encoder_layer(d_model, nhead, dropout, use_local_attn=True)
            for _ in range(num_attn_layers)
        ])
        self.agent_map_attn_layers = nn.ModuleList([
            self._build_encoder_layer(d_model, nhead, dropout, use_local_attn=True)
            for _ in range(num_attn_layers)
        ])

        self.num_out_channels = d_model

    def _build_polyline(self, in_channels, hidden_dim, num_layers, num_pre_layers=1, out_channels=None):
        use_fourier = self.model_cfg.get('USE_FOURIER_POS', False)
        fourier_freqs = self.model_cfg.get('FOURIER_NUM_FREQS', 8)
        if use_fourier:
            return polyline_encoder.PointNetPolylineEncoderWithFourier(
                in_channels=in_channels, hidden_dim=hidden_dim, num_layers=num_layers,
                num_pre_layers=num_pre_layers, out_channels=out_channels,
                use_fourier_pos=True, fourier_freqs=fourier_freqs,
            )
        return polyline_encoder.PointNetPolylineEncoder(
            in_channels=in_channels, hidden_dim=hidden_dim, num_layers=num_layers,
            num_pre_layers=num_pre_layers, out_channels=out_channels,
        )

    def _build_encoder_layer(self, d_model, nhead, dropout, use_local_attn=False):
        return transformer_encoder_layer.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, normalize_before=False, use_local_attn=use_local_attn,
        )

    def _apply_local_attn_batched(self, layers, x, x_mask, x_pos, num_of_neighbors):
        """Apply stacked local-attention layers, batched across center objects.

        Reuses the same KNN logic as MTREncoder.apply_local_attn.
        Args:
            x: (num_center_objects, N, C)
            x_mask: (num_center_objects, N) bool
            x_pos: (num_center_objects, N, 3)
        """
        assert torch.all(x_mask.sum(dim=-1) > 0)
        batch_size, N, d_model = x.shape

        x_stack_full = x.view(-1, d_model)
        x_mask_stack = x_mask.view(-1)
        x_pos_stack_full = x_pos.view(-1, 3)
        batch_idxs_full = torch.arange(batch_size).type_as(x)[:, None].repeat(1, N).view(-1).int()

        x_stack = x_stack_full[x_mask_stack]
        x_pos_stack = x_pos_stack_full[x_mask_stack]
        batch_idxs = batch_idxs_full[x_mask_stack]

        batch_offsets = common_utils.get_batch_offsets(batch_idxs=batch_idxs, bs=batch_size).int()
        batch_cnt = batch_offsets[1:] - batch_offsets[:-1]

        index_pair = knn_utils.knn_batch_mlogk(
            x_pos_stack, x_pos_stack, batch_idxs, batch_offsets, num_of_neighbors
        )

        pos_embedding = position_encoding_utils.gen_sineembed_for_position(
            x_pos_stack[None, :, 0:2], hidden_dim=d_model
        )[0]

        output = x_stack
        for layer in layers:
            output = layer(
                src=output, pos=pos_embedding,
                index_pair=index_pair,
                query_batch_cnt=batch_cnt, key_batch_cnt=batch_cnt,
                index_pair_batch=batch_idxs,
            )

        ret_full_feature = torch.zeros_like(x_stack_full)
        ret_full_feature[x_mask_stack] = output
        return ret_full_feature.view(batch_size, N, d_model)

    def forward(self, batch_dict):
        input_dict = batch_dict['input_dict']
        obj_trajs = input_dict['obj_trajs'].to(self.device)
        obj_trajs_mask = input_dict['obj_trajs_mask'].to(self.device)
        map_polylines = input_dict['map_polylines'].to(self.device)
        map_polylines_mask = input_dict['map_polylines_mask'].to(self.device)

        obj_trajs_last_pos = input_dict['obj_trajs_last_pos'].to(self.device)
        map_polylines_center = input_dict['map_polylines_center'].to(self.device)
        track_index_to_predict = input_dict['track_index_to_predict']

        assert obj_trajs_mask.dtype == torch.bool and map_polylines_mask.dtype == torch.bool

        num_center_objects, num_objects, num_timestamps, _ = obj_trajs.shape
        num_polylines = map_polylines.shape[1]

        # 1. encode polylines
        obj_trajs_in = torch.cat((obj_trajs, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1)
        agent_features = self.agent_polyline_encoder(obj_trajs_in, obj_trajs_mask)  # (Nco, No, C)
        map_features = self.map_polyline_encoder(map_polylines, map_polylines_mask)  # (Nco, Nm, C)

        agent_valid_mask = (obj_trajs_mask.sum(dim=-1) > 0)
        map_valid_mask = (map_polylines_mask.sum(dim=-1) > 0)

        num_neighbors = self.model_cfg.NUM_OF_ATTN_NEIGHBORS

        # 2. factorized local attention — all batched across center objects
        # Pass 1: agent self-attention
        agent_features = self._apply_local_attn_batched(
            self.agent_attn_layers, agent_features, agent_valid_mask,
            obj_trajs_last_pos, num_neighbors
        )

        # Pass 2: map self-attention
        map_features = self._apply_local_attn_batched(
            self.map_attn_layers, map_features, map_valid_mask,
            map_polylines_center, num_neighbors
        )

        # Pass 3: agent-map cross-attention via joint local attention
        # Concatenate agent+map, apply local attention (agents attend to nearby map tokens)
        joint_feature = torch.cat((agent_features, map_features), dim=1)
        joint_mask = torch.cat((agent_valid_mask, map_valid_mask), dim=1)
        joint_pos = torch.cat((obj_trajs_last_pos, map_polylines_center), dim=1)

        joint_feature = self._apply_local_attn_batched(
            self.agent_map_attn_layers, joint_feature, joint_mask,
            joint_pos, num_neighbors
        )

        agent_features = joint_feature[:, :num_objects]
        map_features = joint_feature[:, num_objects:]

        # organize return features (same interface as MTREncoder)
        center_objects_feature = agent_features[torch.arange(num_center_objects), track_index_to_predict]

        batch_dict['center_objects_feature'] = center_objects_feature
        batch_dict['obj_feature'] = agent_features
        batch_dict['map_feature'] = map_features
        batch_dict['obj_mask'] = agent_valid_mask
        batch_dict['map_mask'] = map_valid_mask
        batch_dict['obj_pos'] = obj_trajs_last_pos
        batch_dict['map_pos'] = map_polylines_center

        return batch_dict
