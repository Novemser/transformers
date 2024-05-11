import torch
import copy
import numpy as np
from tqdm import tqdm
import torch.nn as nn

def P_i_j_k(i_index: int, j_index: int, k_index: int, up_proj: torch.Tensor, down_proj: torch.Tensor):
    # return up_proj[j_index][k_index] * down_proj[k_index][i_index]
    return up_proj[k_index][j_index] * down_proj[i_index][k_index]


def generate_act_hats_llama2(
    d_model: int, 
    d_intermediate: int, 
    record_act_result: torch.Tensor,
    up_proj: torch.Tensor, 
    down_proj: torch.Tensor):
    record_act_result = record_act_result.reshape(-1, d_intermediate)
    num_labels = record_act_result.shape[0]
    print(f"num_labels: {num_labels}")
    output = None
    for lable_id in tqdm(range(num_labels), desc="Processing samples:"):
        activation = record_act_result[lable_id]
        assert activation.shape[0] == d_intermediate, "Unexpected recorded activation shape!"
        # calculate a_hat_i_j
        for i_index in tqdm(range(d_model), desc="Processing i_index"):
            for j_index in tqdm(range(d_model), desc="Processing j_index"):
                upper = 0
                lower = 0
                for k_index in tqdm(range(d_intermediate), desc="Processing k_index"):
                    p_i_j_k = P_i_j_k(i_index=i_index, j_index=j_index, k_index=k_index, up_proj=up_proj, down_proj=down_proj)
                    upper += activation[k_index] * p_i_j_k
                    lower += p_i_j_k
                a_hat_i_j = (upper / lower).unsqueeze(-1)
                if output == None:
                    output = a_hat_i_j
                else:
                    output = torch.cat((output, a_hat_i_j), dim=0)
    return output.reshape(num_labels, d_model, d_model)