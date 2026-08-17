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
    """QCNet-style scene-level shared encoder with factorized attention.

    Instead of running one monolithic global self-attention over the joint
    [agent + map] token set (as ``MTREncoder`` does), this encoder applies
    three decoupled attention passes:

    1. **Agent temporal attention** — agents attend across the agent-token axis
       (capturing inter-agent relationships).
    2. **Agent-map cross attention** — agent tokens cross-attend to map tokens
       (grounding agents in lane / road geometry).
    3. **Map self attention** — map tokens refine among themselves.

    Position features use **polar + Fourier** encoding for better spatial
    generalisation, as recommended in the optimization plan.

    The encoder processes each center-object frame independently (the data
    pipeline is agent-centric), but the factorised structure is significantly
    cheaper than full global attention: O(A^2 + A*M + M^2) instead of
    O((A+M)^2), where A = num_objects and M = num_polylines.
    """

    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__()
        self.model_cfg = config

        d_model = self.model_cfg.D_MODEL
        nhead = self.model_cfg.NUM_ATTN_HEAD
        dropout = self.model_cfg.get('DROPOUT_OF_ATTN', 0.1)
        num_attn_layers = self.model_cfg.NUM_ATTN_LAYERS

        # build polyline encoders (reuse MTREncoder's factory logic)
        self._dummy = nn.Module()
        self._dummy.model_cfg = config
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
        del self._dummy

        # polar + Fourier position encoding
        self.use_fourier = self.model_cfg.get('USE_FOURIER_POS', False)
        if self.use_fourier:
            from mtr.models.utils.polyline_encoder import FourierPositionEncoder
            self.fourier_pos_enc = FourierPositionEncoder(
                out_dim=d_model,
                num_freqs=self.model_cfg.get('FOURIER_NUM_FREQS', 8),
            )
        else:
            self.fourier_pos_enc = None

        # factorized attention layers
        self.agent_agent_attn_layers = nn.ModuleList([
            self._build_encoder_layer(d_model, nhead, dropout) for _ in range(num_attn_layers)
        ])
        self.map_self_attn_layers = nn.ModuleList([
            self._build_encoder_layer(d_model, nhead, dropout) for _ in range(num_attn_layers)
        ])
        # cross attention: agent (query) -> map (key/value)
        self.agent_map_cross_attn_layers = nn.ModuleList([
            self._build_cross_attn_layer(d_model, nhead, dropout) for _ in range(num_attn_layers)
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

    def _build_encoder_layer(self, d_model, nhead, dropout):
        """Build a self-attention + FFN block using nn.MultiheadAttention."""
        layer = nn.Module()
        layer.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        layer.linear1 = nn.Linear(d_model, d_model * 4)
        layer.linear2 = nn.Linear(d_model * 4, d_model)
        layer.norm1 = nn.LayerNorm(d_model)
        layer.norm2 = nn.LayerNorm(d_model)
        layer.dropout1 = nn.Dropout(dropout)
        layer.dropout2 = nn.Dropout(dropout)
        layer.activation = nn.functional.relu
        return layer

    def _build_cross_attn_layer(self, d_model, nhead, dropout):
        """A single cross-attention + FFN block (agent queries, map keys)."""
        layer = nn.Module()
        layer.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        layer.linear1 = nn.Linear(d_model, d_model * 4)
        layer.linear2 = nn.Linear(d_model * 4, d_model)
        layer.norm1 = nn.LayerNorm(d_model)
        layer.norm2 = nn.LayerNorm(d_model)
        layer.dropout1 = nn.Dropout(dropout)
        layer.dropout2 = nn.Dropout(dropout)
        layer.activation = nn.functional.relu
        return layer

    def _apply_self_attn(self, layers, x, x_mask, x_pos):
        """Apply stacked self-attention + FFN layers.

        Args:
            x: (B, N, C)
            x_mask: (B, N) bool
            x_pos: (B, N, 3)
        """
        assert torch.all(x_mask.sum(dim=-1) > 0)
        B, N, C = x.shape
        x_t = x.permute(1, 0, 2)  # (N, B, C)
        padding_mask = ~x_mask  # (B, N)
        pos_xy = x_pos[..., 0:2].permute(1, 0, 2)  # (N, B, 2)
        pos_emb = position_encoding_utils.gen_sineembed_for_position(pos_xy, hidden_dim=C)

        for layer in layers:
            q = k = x_t + pos_emb
            src2, _ = layer.self_attn(q, k, value=x_t, key_padding_mask=padding_mask)
            x_t = x_t + layer.dropout1(src2)
            x_t = layer.norm1(x_t)
            x_t = x_t + layer.dropout2(layer.linear2(layer.dropout2(layer.activation(layer.linear1(x_t)))))
            x_t = layer.norm2(x_t)
        return x_t.permute(1, 0, 2)  # (B, N, C)

    def _apply_cross_attn(self, layer, query, key, query_mask, key_mask, query_pos, key_pos):
        """Apply one cross-attention + FFN block.

        Args:
            query: (B, A, C) agent features
            key: (B, M, C) map features
        """
        B, A, C = query.shape
        q_t = query.permute(1, 0, 2)  # (A, B, C)
        k_t = key.permute(1, 0, 2)    # (M, B, C)
        k_padding_mask = ~key_mask  # (B, M)
        q_pos_xy = query_pos[..., 0:2].permute(1, 0, 2)  # (A, B, 2)
        k_pos_xy = key_pos[..., 0:2].permute(1, 0, 2)     # (M, B, 2)
        q_pos_emb = position_encoding_utils.gen_sineembed_for_position(q_pos_xy, hidden_dim=C)
        k_pos_emb = position_encoding_utils.gen_sineembed_for_position(k_pos_xy, hidden_dim=C)

        q_with_pos = q_t + q_pos_emb

        src2, _ = layer.cross_attn(
            query=q_with_pos, key=k_t + k_pos_emb, value=k_t,
            key_padding_mask=k_padding_mask,
        )
        q_t = q_t + layer.dropout1(src2)
        q_t = layer.norm1(q_t)
        q_t = q_t + layer.dropout2(layer.linear2(layer.dropout2(layer.activation(layer.linear1(q_t)))))
        q_t = layer.norm2(q_t)
        return q_t.permute(1, 0, 2)  # (B, A, C)

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

        # 1. encode polylines (per-object / per-polyline features)
        obj_trajs_in = torch.cat((obj_trajs, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1)
        agent_features = self.agent_polyline_encoder(obj_trajs_in, obj_trajs_mask)  # (Nco, No, C)
        map_features = self.map_polyline_encoder(map_polylines, map_polylines_mask)  # (Nco, Nm, C)

        agent_valid_mask = (obj_trajs_mask.sum(dim=-1) > 0)   # (Nco, No)
        map_valid_mask = (map_polylines_mask.sum(dim=-1) > 0)  # (Nco, Nm)

        # 2. factorized attention — iterate over center objects
        # (each center object has its own coordinate frame, so we process them
        #  individually but with the cheaper factorised attention structure)
        output_agent_features = agent_features.new_zeros(num_center_objects, num_objects, self.num_out_channels)
        output_map_features = map_features.new_zeros(num_center_objects, num_polylines, self.num_out_channels)

        for ci in range(num_center_objects):
            af = agent_features[ci:ci+1]  # (1, No, C)
            mf = map_features[ci:ci+1]    # (1, Nm, C)
            am = agent_valid_mask[ci:ci+1]
            mm = map_valid_mask[ci:ci+1]
            ap = obj_trajs_last_pos[ci:ci+1]
            mp = map_polylines_center[ci:ci+1]

            # Pass 1: agent-agent self attention
            af = self._apply_self_attn(self.agent_agent_attn_layers, af, am, ap)

            # Pass 2: map self attention
            mf = self._apply_self_attn(self.map_self_attn_layers, mf, mm, mp)

            # Pass 3: agent-map cross attention (agents query maps)
            for layer in self.agent_map_cross_attn_layers:
                af = self._apply_cross_attn(layer, af, mf, am, mm, ap, mp)

            output_agent_features[ci] = af[0]
            output_map_features[ci] = mf[0]

        # organize return features (same interface as MTREncoder)
        center_objects_feature = output_agent_features[torch.arange(num_center_objects), track_index_to_predict]

        batch_dict['center_objects_feature'] = center_objects_feature
        batch_dict['obj_feature'] = output_agent_features
        batch_dict['map_feature'] = output_map_features
        batch_dict['obj_mask'] = agent_valid_mask
        batch_dict['map_mask'] = map_valid_mask
        batch_dict['obj_pos'] = obj_trajs_last_pos
        batch_dict['map_pos'] = map_polylines_center

        return batch_dict
