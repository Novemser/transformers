import os
import torch
import copy
import numpy as np
from tqdm import tqdm
import torch.nn as nn

def P_i_j_k(i_index: int, j_index: int, k_index: int, up_proj: torch.Tensor, down_proj: torch.Tensor):
    return up_proj[j_index][k_index] * down_proj[k_index][i_index]

def generate_act_hats_llama2(
    d_model: int, 
    d_intermediate: int, 
    record_act_result: torch.Tensor,
    up_proj: torch.Tensor, 
    down_proj: torch.Tensor,
    num_samples_per_file: int='100',
    activation_recorded_res_path: str='./',
    file_name_prefix: str='labels',
    max_samples: int=500,
    save_result: bool=True):
    record_act_result = record_act_result.reshape(-1, d_intermediate)
    num_labels = record_act_result.shape[0]
    print(f"num_labels: {num_labels}")
    output = None
    denominator = torch.matmul(up_proj, down_proj).float() + 1e-8
    part_index = 0
    with torch.no_grad():
        for lable_id in tqdm(range(num_labels), desc="Processing samples (batch size=1):"):
            activation = record_act_result[lable_id].to(device=up_proj.device)
            assert activation.shape[0] == d_intermediate, "Unexpected recorded activation shape!"
            # calculate a_hat_i_j
            numerator = torch.matmul(activation * up_proj, down_proj).float()
            a_hat_i_j = (numerator / denominator).half()
            assert a_hat_i_j.isnan().sum() == 0
            if output == None:
                output = a_hat_i_j
            else:
                output = torch.cat((output, a_hat_i_j), dim=0)
                if (output.shape[0] / d_model) >= num_samples_per_file:
                    # save partial result
                    if save_result:
                        torch.save(output.half().reshape(num_samples_per_file, d_model * d_model), os.path.join(
                            activation_recorded_res_path, 
                            f"{file_name_prefix}_part_{part_index}.pt"))
                    else:
                        print(output)
                    output = None
                    part_index += 1
                if max_samples <= (num_samples_per_file * part_index):
                    print(f"Collected {num_samples_per_file * part_index} samples, exiting...")
                    return

        if output != None:
            if save_result:
                torch.save(output.half().reshape(part_index % num_samples_per_file + 1, d_model * d_model), os.path.join(
                    activation_recorded_res_path, 
                    f"{file_name_prefix}_part_{part_index}.pt"))
            else:
                print(output)