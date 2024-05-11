import torch
import copy
import numpy as np
from tqdm import tqdm
import torch.nn as nn

def P_i_j_k(i_index: int, j_index: int, k_index: int, up_proj: torch.Tensor, down_proj: torch.Tensor):
    return up_proj[j_index][k_index] * down_proj[k_index][i_index]
    # return up_proj[k_index][j_index] * down_proj[i_index][k_index]


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
        activation = record_act_result[lable_id].to(device=up_proj.device)
        assert activation.shape[0] == d_intermediate, "Unexpected recorded activation shape!"
        # calculate a_hat_i_j
        for i_index in tqdm(range(d_model), desc="Processing i_index"):
            for j_index in tqdm(range(d_model), desc="Processing j_index"):
                u_jth_row = up_proj[j_index, :]#.unsqueeze(0)
                o_ith_col = down_proj[:, i_index]#.unsqueeze(1)
                mul = u_jth_row * o_ith_col
                c_ij = mul.sum()
                numerator = (activation * mul).sum()
                a_hat_i_j = (numerator / c_ij).unsqueeze(-1)
                if output == None:
                    output = a_hat_i_j
                else:
                    output = torch.cat((output, a_hat_i_j), dim=0)
    return output.reshape(num_labels, d_model, d_model)