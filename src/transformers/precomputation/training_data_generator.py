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
    record_input_result: torch.Tensor,
    num_samples_per_file: int=100,
    activation_recorded_res_path: str='./',
    file_name_prefix: str='labels',
    max_samples: int=500,
    save_result: bool=True):
    record_act_result = record_act_result.reshape(-1, d_intermediate)
    record_input_result = record_input_result.reshape(-1, d_model)
    num_labels = record_act_result.shape[0]
    print(f"num_labels: {num_labels}")
    output = None
    C = torch.matmul(up_proj, down_proj).float().transpose(0, 1) + 1e-8 # The C matrix, D * D
    part_index = 0
    with torch.no_grad():
        for lable_id in tqdm(range(num_labels), desc="Processing samples (batch size=1):"):
            activation = record_act_result[lable_id].to(device=up_proj.device) # 1 * I
            record_input = record_input_result[lable_id].to(device=up_proj.device) # 1 * D
            assert activation.shape[0] == d_intermediate, "Unexpected recorded activation shape!"
            assert record_input.shape[0] == d_model, "Unexpected recorded record_input shape!"
            # calculate a_hat
            AUO = torch.matmul(activation * up_proj, down_proj).float().transpose(0, 1) # D * D
            a_hat = (AUO / C) # D * D
            XC = (record_input.float() * C.transpose(0, 1)).sum(dim=1) # The X*C matrix, 1 * D
            Xa_hatC = (record_input.float() * a_hat.float() * C).sum(dim=1) # 1 * D
            a_hat_hat = Xa_hatC / XC # 1 * D
            
            assert a_hat_hat.isnan().sum() == 0
            if output == None:
                output = a_hat_hat
            else:
                output = torch.cat((output, a_hat_hat), dim=0)
                if (output.shape[0] / d_model) >= num_samples_per_file:
                    # save partial result
                    if save_result:
                        torch.save(output.reshape(num_samples_per_file, d_model), os.path.join(
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
                torch.save(output.reshape(-1, d_model), 
                           os.path.join(activation_recorded_res_path, f"{file_name_prefix}_part_{part_index}.pt"))
            else:
                print(output)