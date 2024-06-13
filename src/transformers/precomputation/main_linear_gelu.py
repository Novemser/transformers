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

def get_data(samples_to_learn, range_min=-0.5, range_max=0.5, layer_idx=0):
    # 方案1. 自己生成某个range的数据
    query = torch.FloatTensor(samples_to_learn, CONFIG["intermediate"]).uniform_(range_min, range_max).cuda()
    act_fun = GELUActivation()
    return query, act_fun(query)
    # 方案2. 根据记录的mlp数据，算某个linear range的数据
    # model = init_model()
    # model.eval()
    # act_fun = GELUActivation()
    # _, act_inputs = get_mlp_and_act_inputs(model=model, layer_idx=layer_idx)
    # act_inputs = act_inputs[:samples_to_learn,:].float().cuda()
    # mask = ((act_inputs >= range_min) * (act_inputs < range_max))
    # act_inputs = act_inputs * mask
    # for act_input in act_inputs:
    #     act_input
    # act_inputs = act_inputs[(act_inputs * mask).nonzero()]
    # return act_inputs, act_fun(act_inputs)

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
        default=100,
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
    step = 0.25
    start = -3
    end = 3
    for range_to_learn in tqdm(gen_range_by_step(start, end, step), desc=f"Training linear GELU with ({start},{end},{step})"):
        # range_to_learn = (-0.75, -0.5)
        query, labels = get_data(CONFIG['samples_to_learn'], layer_idx=args.L, range_min=range_to_learn[0], range_max=range_to_learn[1])

        train_loader, test_loader = create_dataset(query, labels, args)

        query_layer = torch.nn.Sequential(
            torch.nn.Linear(CONFIG["intermediate"], args.D, bias=True, dtype=torch.float32),
            torch.nn.Linear(args.D, CONFIG["intermediate"], bias=True, dtype=torch.float32)
        )

        print("Start Training")
        best_model, eval_result = train(
            query_layer,  train_loader, test_loader, args, device, verbal=True
        )

        dir = f"/root/autodl-tmp/predictors/linear-gelu-predictor"
        file = f"model_range_{range_to_learn[0]}_{range_to_learn[1]}.pt"
        if not os.path.exists(dir):
            os.makedirs(dir)
        torch.save(best_model, os.path.join(dir, file))

if __name__ == "__main__":
    main()