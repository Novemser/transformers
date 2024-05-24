import os
import torch
import copy
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import math

def P_i_j_k(i_index: int, j_index: int, k_index: int, up_proj: torch.Tensor, down_proj: torch.Tensor):
    return up_proj[j_index][k_index] * down_proj[k_index][i_index]

# 解一元二次方程
def solve_quadratic(a :torch.tensor, b :torch.tensor, c :torch.tensor):
    # 计算判别式
    discriminant = b**2 - 4*a*c
    # 初始化根的Tensor
    root1 = torch.zeros_like(a, dtype=torch.cfloat)
    root2 = torch.zeros_like(a, dtype=torch.cfloat)
    
    # 计算实根
    real_discriminant_mask = discriminant >= 0
    sqrt_discriminant_real = torch.sqrt(discriminant[real_discriminant_mask])
    root1[real_discriminant_mask] = ((-b[real_discriminant_mask] + sqrt_discriminant_real) / (2*a[real_discriminant_mask])).to(torch.cfloat)
    root2[real_discriminant_mask] = ((-b[real_discriminant_mask] - sqrt_discriminant_real) / (2*a[real_discriminant_mask])).to(torch.cfloat)
    
    # 计算复数根
    complex_discriminant_mask = discriminant < 0
    sqrt_discriminant_complex = torch.sqrt(-discriminant[complex_discriminant_mask])
    real_part = -b[complex_discriminant_mask] / (2*a[complex_discriminant_mask])
    imaginary_part = sqrt_discriminant_complex / (2*a[complex_discriminant_mask])
    root1[complex_discriminant_mask] = torch.complex(real_part, imaginary_part)
    root2[complex_discriminant_mask] = torch.complex(real_part, -imaginary_part)

    return root1, root2

def generate_x_hats_llama2(
    d_model: int, 
    d_intermediate: int,
    gate_proj: torch.Tensor, 
    up_proj: torch.Tensor, 
    down_proj: torch.Tensor,
    record_inputs: torch.Tensor,
    save_result: bool=True,
    batch_size: int=1,
    linear_silu: tuple=(torch.tensor(0.48, device='cuda'), torch.tensor(0.0, device='cuda'))):    
    record_inputs = record_inputs.reshape(-1, batch_size, d_model).cuda()
    x_squared_coefficients = torch.zeros(1, d_model, device='cuda')
    x_coefficients = torch.zeros(1, d_model, device='cuda')
    for k in tqdm(range(d_intermediate), desc="Calculating coefficient matrix"):
        g_sum = gate_proj[:, k].sum()
        u_sum = up_proj[:, k].sum()
        x_squared_coefficients += linear_silu[0] * g_sum * u_sum * down_proj[k, :]
        x_coefficients += linear_silu[1] * u_sum * down_proj[k, :]

    assert x_squared_coefficients.shape == (1, d_model)
    assert x_coefficients.shape == (1, d_model)
    res_labels = None
    silu = torch.nn.SiLU()
    with torch.no_grad():
        for batch in tqdm(record_inputs):
            assert batch.shape == torch.Size((batch_size, d_model))
            output_actual = torch.mm(silu(torch.mm(batch, gate_proj)) * torch.mm(batch, up_proj), down_proj)
            C = -output_actual
            root1, root2 = solve_quadratic(x_squared_coefficients, x_coefficients, C)
            label = torch.cat((root1, root2), dim=1)
            if res_labels == None:
                res_labels = label
            else:
                res_labels = torch.cat((res_labels, label), dim=0)
    return res_labels, torch.cat((x_squared_coefficients, x_coefficients), dim=1)

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