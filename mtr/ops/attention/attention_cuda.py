"""
Pure PyTorch fallback for attention_cuda (CPU-compatible).
Provides the same interface as the CUDA extension but uses torch native ops.
"""
import torch


def attention_weight_computation_wrapper(
    b, total_query_num, local_size, total_key_num, nhead, hdim,
    query_batch_cnt, key_batch_cnt, index_pair_batch,
    index_pair, query_features, key_features, output):
    """Compute attention weights: dot product of query and key features for each index pair.
    
    output: (total_query_num, local_size, nhead) - modified in-place
    """
    # query_features: (total_query_num, nhead, hdim)
    # key_features: (total_key_num, nhead, hdim)
    # index_pair: (total_query_num, local_size) - key indices, -1 means invalid
    
    # Gather key features: (total_query_num, local_size, nhead, hdim)
    valid_mask = index_pair >= 0  # (total_query_num, local_size)
    safe_index_pair = index_pair.clamp(min=0)  # avoid negative indexing
    gathered_keys = key_features[safe_index_pair]  # (total_query_num, local_size, nhead, hdim)
    
    # Compute dot product: (total_query_num, local_size, nhead)
    # query_features: (total_query_num, nhead, hdim) -> (total_query_num, 1, nhead, hdim)
    dot = (gathered_keys * query_features.unsqueeze(1)).sum(dim=-1)  # (total_query_num, local_size, nhead)
    
    # Zero out invalid entries
    dot[~valid_mask.unsqueeze(-1).expand_as(dot)] = 0.0
    output.copy_(dot)


def attention_weight_computation_grad_wrapper(
    b, total_query_num, local_size, total_key_num, nhead, hdim,
    query_batch_cnt, key_batch_cnt, index_pair_batch,
    index_pair, query_features, key_features,
    grad_out, grad_query_features, grad_key_features):
    """Backward for attention weight computation."""
    valid_mask = index_pair >= 0  # (total_query_num, local_size)
    safe_index_pair = index_pair.clamp(min=0)
    gathered_keys = key_features[safe_index_pair]  # (total_query_num, local_size, nhead, hdim)
    
    # grad_out: (total_query_num, local_size, nhead)
    grad_out_expanded = grad_out.unsqueeze(-1)  # (total_query_num, local_size, nhead, 1)
    
    # grad_query = sum over local_size of grad_out * key
    grad_q = (grad_out_expanded * gathered_keys).sum(dim=1)  # (total_query_num, nhead, hdim)
    grad_query_features.copy_(grad_q)
    
    # grad_key = scatter_add grad_out * query back to key positions
    grad_k = grad_out_expanded * query_features.unsqueeze(1)  # (total_query_num, local_size, nhead, hdim)
    grad_k[~valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(grad_k)] = 0.0
    
    # Scatter add back to key features
    grad_key_features.zero_()
    for i in range(local_size):
        idx = safe_index_pair[:, i]  # (total_query_num,)
        valid = valid_mask[:, i]
        if valid.any():
            grad_key_features.index_add_(
                0, idx[valid], grad_k[valid, i]
            )


def attention_value_computation_wrapper(
    b, total_query_num, local_size, total_key_num, nhead, hdim,
    query_batch_cnt, key_batch_cnt, index_pair_batch,
    index_pair, attn_weight, value_features, output):
    """Compute attention values: weighted sum of value features.
    
    output: (total_query_num, nhead, hdim) - modified in-place
    """
    # attn_weight: (total_query_num, local_size, nhead)
    # value_features: (total_key_num, nhead, hdim)
    # index_pair: (total_query_num, local_size)
    
    valid_mask = index_pair >= 0
    safe_index_pair = index_pair.clamp(min=0)
    gathered_values = value_features[safe_index_pair]  # (total_query_num, local_size, nhead, hdim)
    
    # Weighted sum: (total_query_num, nhead, hdim)
    weighted = gathered_values * attn_weight.unsqueeze(-1)  # (total_query_num, local_size, nhead, hdim)
    weighted[~valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(weighted)] = 0.0
    result = weighted.sum(dim=1)  # (total_query_num, nhead, hdim)
    output.copy_(result)


def attention_value_computation_grad_wrapper(
    b, total_query_num, local_size, total_key_num, nhead, hdim,
    query_batch_cnt, key_batch_cnt, index_pair_batch,
    index_pair, attn_weight, value_features,
    grad_out, grad_attn_weight, grad_value_features):
    """Backward for attention value computation."""
    valid_mask = index_pair >= 0
    safe_index_pair = index_pair.clamp(min=0)
    gathered_values = value_features[safe_index_pair]  # (total_query_num, local_size, nhead, hdim)
    
    # grad_attn_weight = sum over hdim of grad_out * value
    grad_aw = (grad_out.unsqueeze(1) * gathered_values).sum(dim=-1)  # (total_query_num, local_size, nhead)
    grad_aw[~valid_mask.unsqueeze(-1).expand_as(grad_aw)] = 0.0
    grad_attn_weight.copy_(grad_aw)
    
    # grad_value = scatter_add grad_out * attn_weight back
    grad_v = grad_out.unsqueeze(1) * attn_weight.unsqueeze(-1)  # (total_query_num, local_size, nhead, hdim)
    grad_v[~valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(grad_v)] = 0.0
    
    grad_value_features.zero_()
    for i in range(local_size):
        idx = safe_index_pair[:, i]
        valid = valid_mask[:, i]
        if valid.any():
            grad_value_features.index_add_(
                0, idx[valid], grad_v[valid, i]
            )


# V2 variants (same interface, identical computation for CPU fallback)
attention_weight_computation_wrapper_v2 = attention_weight_computation_wrapper
attention_weight_computation_grad_wrapper_v2 = attention_weight_computation_grad_wrapper
attention_value_computation_wrapper_v2 = attention_value_computation_wrapper
attention_value_computation_grad_wrapper_v2 = attention_value_computation_grad_wrapper
