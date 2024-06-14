import os
import torch
import numpy as np
import argparse
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from trainer_mlp_scaler import train
import random
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.activations import GELUActivation
from transformers.prune.sparsity_util import gen_range_by_step
from tqdm import tqdm


MODEL_CHOICES = ['7b']
DATA_CHOICES = ['piqa']
CONFIG = {
    'num_layer': 32,
    'd_model': 4544,
    'intermediate': 18176,
    'samples_to_learn': 30000,
    'samples_per_file': 50000
}

def init_model(ckpt_name: str='tiiuae/falcon-7b'):
    return AutoModelForCausalLM.from_pretrained(ckpt_name, torch_dtype="auto", device_map="auto").cuda()

def load_recorded_mpl_input(path: str, layer_idx: str) -> torch.Tensor:
    return torch.load(os.path.join(path, f"{layer_idx}_mlp_mlp_input.pt")).cuda().bfloat16()

def get_mlp_and_act_inputs(model, layer_idx, recorded_mpl_input_path='/root/autodl-tmp/instrumentation_output_dir/falcon-7b/piqa') -> torch.Tensor:
    recorded_mpl_input = load_recorded_mpl_input(recorded_mpl_input_path, layer_idx)
    with torch.no_grad():
        return recorded_mpl_input.cpu().float(), model.transformer.h[layer_idx].mlp.dense_h_to_4h(recorded_mpl_input.cuda()).cpu().float()

class BasicDataset(Dataset):
    def __init__(self, X, Y, n, train):
        self.X = X
        self.Y = Y 
        self.n = n
        self.train = train

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.train:
            x = torch.Tensor(self.X[idx])
            y = torch.Tensor(self.Y[idx])
        else:
            x = torch.Tensor(self.X[-idx])
            y = torch.Tensor(self.Y[-idx])
        if y.sum()== 0:
            print("all zero y")
            exit()
        return x, y
    
def init_model(ckpt_name: str='tiiuae/falcon-7b'):
    return AutoModelForCausalLM.from_pretrained(ckpt_name, torch_dtype="auto", device_map="auto").cuda()

def get_data(samples_to_learn, model, layer_idx, linear_ranges):
    mlp_input, act_input = get_mlp_and_act_inputs(model, layer_idx)
    act_input = act_input[:samples_to_learn,:].cuda()
    mlp_input = mlp_input[:samples_to_learn,:].cuda()
    # 如果有L个区间，I个神经元
    # 1. 确定给每个神经元选择的线性区间是什么。结果是一个1*I的矩阵，每个元素有L+1个可能性(0~L, 0表示不在任何线性区间里)
    max = None
    # initially, all neurals out of linear scope
    choosen_linear_range = torch.zeros(1, CONFIG["intermediate"], device=act_input.device, dtype=torch.int16)
    choosen_linear_range_min = torch.ones(1, CONFIG["intermediate"], device=act_input.device, dtype=torch.float) * 10000
    choosen_linear_range_max = torch.ones(1, CONFIG["intermediate"], device=act_input.device, dtype=torch.float) * -10000
    for _range_idx, _range in enumerate(linear_ranges):
        vals_in_range = ((act_input >= _range[0]) * (act_input < _range[1])).sum(dim=0) / act_input.shape[0]
        last_max = max if max != None else 0
        max = vals_in_range if max == None else torch.maximum(max, vals_in_range)
        # the part whose neural has changed to the current linear range
        diff_mask = (max > last_max)
        # erase the diff and set diff with current range index
        unchanged_mask = (diff_mask == False)
        choosen_linear_range *= unchanged_mask
        choosen_linear_range += diff_mask.to(torch.int16) * (_range_idx + 1)
        # adjust neural range accordingly. Neurals out of linear scope has empty min/max range; other neruals have min/max linear range
        choosen_linear_range_min *= unchanged_mask
        choosen_linear_range_min += diff_mask.to(torch.float) * _range[0]
        choosen_linear_range_max *= unchanged_mask
        choosen_linear_range_max += diff_mask.to(torch.float) * _range[1]

    # 2. 对于给定的input X，确定I个神经元分别是否落在了自己被分到的线性区间里。生成结果是一个1*I的矩阵，每个元素是0/1，即label
    in_range_mask = (act_input < choosen_linear_range_max) * (act_input >= choosen_linear_range_min)
    return mlp_input, in_range_mask.float()

def create_dataset(query, labels, args):

    total = len(query)
    num_train = int(0.95 * total)
    num_test = int(0.05 * total)

    print(f"Query shape: {query.shape}, Label shape: {labels.shape}")
    print(f"# training data: {num_train}, # test data: {num_test}")

    train_ds = BasicDataset(query, labels, num_train, True)
    test_ds = BasicDataset(query, labels, num_test, False)

    train_dataloader = DataLoader(
        train_ds, args.batch_size, shuffle=True, num_workers=0
    )
    test_dataloader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=0)
    return train_dataloader, test_dataloader


def main():
    parser = argparse.ArgumentParser(description="Linear Silu Predictor")
    parser.add_argument("--model", type=str, default="7b", choices = MODEL_CHOICES)
    parser.add_argument("--dataset", type=str, default="piqa", choices = DATA_CHOICES)
    parser.add_argument(
        "--L",
        type=int,
        default=9,
        help="which layer",
    )
    parser.add_argument(
        "--D",
        type=int,
        default=1000,
        help="low rank dimension",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate",
    )
    args = parser.parse_args()

    print(args)
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    step = 1
    start = -3
    end = 0
    linear_ranges = gen_range_by_step(start, end, step)
    model = init_model()
    for layer_idx in tqdm(range(CONFIG["num_layer"]), desc=f"Training linear neural predictor with ({start},{end},{step})"):
        print(f"Training layer_idx {layer_idx}")
        query, labels = get_data(CONFIG['samples_to_learn'], model, layer_idx, linear_ranges)

        train_loader, test_loader = create_dataset(query, labels, args)

        query_layer = torch.nn.Sequential(
            torch.nn.Linear(CONFIG["d_model"], args.D, bias=True, dtype=torch.float32),
            nn.ReLU(),
            torch.nn.Linear(args.D, CONFIG["intermediate"], bias=True, dtype=torch.float32)
        )
        
        dir = f"/root/autodl-tmp/predictors/linear-neural-predictor"
        file = f"model_layer_idx_{layer_idx}.pt"
        if not os.path.exists(dir):
            os.makedirs(dir)

        if os.path.exists(os.path.join(dir, file)):
            print("Loading previous checkpoint")
            query_layer.load_state_dict(torch.load(os.path.join(dir, file)))

        print(f"Start Training layer {layer_idx}")
        best_model, eval_result = train(
            query_layer,  train_loader, test_loader, args, device, verbal=True, classification=True
        )

        torch.save(best_model, os.path.join(dir, file))

if __name__ == "__main__":
    main()