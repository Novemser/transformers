import os
import torch
import copy
import numpy as np
from tqdm import tqdm
import torch.nn as nn

def P_i_j_k(i_index: int, j_index: int, k_index: int, up_proj: torch.Tensor, down_proj: torch.Tensor):
    return up_proj[j_index][k_index] * down_proj[k_index][i_index]

def generate_x_hats_llama2(
    d_model: int, 
    gate_proj: torch.Tensor, 
    up_proj: torch.Tensor, 
    down_proj: torch.Tensor,
    record_inputs: torch.Tensor,
    save_result: bool=True,
    batch_size: int=10):
    H = gate_proj * up_proj
    V = torch.mm(H, down_proj)
    
    x_squared_coefficients = V.diag()

    # 非对角线元素给出交叉项系数
    cross_terms_sum = 0
    # cross_terms_coefficients = {}
    for i in tqdm(range(d_model), desc="Processing cross_terms_coefficients"):
        for j in range(i + 1, d_model):
            coefficient = V[i, j] + V[j, i]
            # cross_terms_coefficients[f'x{i+1}x{j+1}'] = coefficient
            cross_terms_sum += coefficient
    
    record_inputs = record_inputs.reshape(-1, batch_size, d_model).cuda()
    res = None
    with torch.no_grad():
        for batch in tqdm(record_inputs):
            assert batch.shape == torch.Size((batch_size, d_model))
            output_actual = torch.mm(torch.mm(batch, gate_proj) * torch.mm(batch, up_proj), down_proj)
            # 减去平方项的和
            output_actual -= torch.mm(batch * batch, x_squared_coefficients.unsqueeze(1))
            label = output_actual / cross_terms_sum
            assert label.shape == torch.Size((batch_size, d_model))
            if save_result and res == None:
                res = label
            elif res != None:
                res = torch.cat((res, label), dim=0)
    return res

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