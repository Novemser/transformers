import json
import numpy as np
import torch
from torch import nn
from transformers.utils import logging
logger = logging.get_logger(__name__)

# TOOD: The code is reused from the clip benchmark, we need to refactor the code to make it as a dependened package
def sparsify_matrix_for_FC_layer(hidden_states, sparsity_percentage, dim_of_neurals, enable_random_sparsity_selection=False):
    topk = int(sparsity_percentage * dim_of_neurals)
    # assert len(hidden_states.size()) == 3
    _logit = hidden_states.reshape(-1, dim_of_neurals).float()
    if enable_random_sparsity_selection:
        _top_indices = torch.randint(0, dim_of_neurals, (_logit.size()[0], topk), device='cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        _, _top_indices = _logit.topk(topk, dim=1)
    _top_k_indices = _top_indices[:, :topk]
    mask = torch.zeros_like(_logit).scatter(1, _top_k_indices, 1).bool().half()
    hidden_states = hidden_states * mask.reshape(hidden_states.size())
    return hidden_states

def mask_attention_result(hidden_states: torch.Tensor, sparsity_percentage, num_attention_heads):
    bsz, num_head, token_length, head_size = hidden_states.size()

    with torch.no_grad():
        topk = int(sparsity_percentage * num_attention_heads)
        # Calculate the Matrix L2-norm (Frobenius norm) of the attention heads (based on the last two dimentions)
        _logit = torch.linalg.norm(hidden_states, dim=(-1, -2), ord = 'fro')

        # for each element in a batch, select the K most activated attention heads
        _, _top_k_indices = _logit.topk(int(topk), dim=1)
        _head_mask = torch.zeros(
            bsz,
            num_head,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ).scatter_(1, _top_k_indices, 1)
        # Mask unactivated attention heads
        _head_mask = _head_mask.unsqueeze(-1).unsqueeze(-1).expand_as(hidden_states)
    
    return hidden_states * _head_mask

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def prune_by_weight_importances(
        weight: torch.tensor,
        weight_importances: list[torch.tensor],
        pruning_percentage) -> torch.tensor:
    with torch.no_grad():
        original_size = weight.size()
        # flatten the weight mask/weight_importance matrix for easier processing
        weight = weight.view(-1,)
        mask = torch.ones_like(weight)
        for weight_importance in weight_importances:
            weight_importance = weight_importance.view(-1,)
            topk = int((1 - pruning_percentage) * weight_importance.shape[0])
            preserved_indices = weight_importance.topk(topk).indices
            mask *= torch.zeros_like(weight).scatter(0, preserved_indices, 1).bool().half()
        
        return (weight * mask).view(original_size), 100 * (mask.bool().sum().cpu().item()) / topk

def prune_by_weight_importances_based_on_threshold(
        weight: torch.tensor,
        weight_importances: list[torch.tensor],
        threshold_percentage) -> torch.tensor:
    with torch.no_grad():
        original_size = weight.size()
        # flatten the weight mask/weight_importance matrix for easier processing
        weight = weight.view(-1,)
        mask = torch.ones_like(weight)
        for weight_importance in weight_importances:
            weight_importance = weight_importance.view(-1,)
            threshold = weight_importance.max() * threshold_percentage
            preserved_weights = weight_importance > threshold
            mask *= preserved_weights
        
        return (weight * mask).view(original_size), 100 * (mask.bool().sum().cpu().item()) / mask.numel()

def prune_by_column_importances(
        weight: torch.tensor, 
        weight_importances: torch.tensor,
        pruning_percentage,
        reverse_order=False) -> torch.tensor:
    pass


PRUNING_FUNC_MAP = {
    'weight': prune_by_weight_importances,
    'weight_threshold': prune_by_weight_importances_based_on_threshold,
    'column': prune_by_column_importances
}
task_names = ['copa', 'lambada_openai', 'piqa', 'mmlu', 'gsm8k', 'arc_challenge']
RECORDED_TASKS = task_names