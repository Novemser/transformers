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

DATA = {
    "7b": {
        "piqa": "/root/autodl-tmp/mlp_activation/Llama-2-7b-chat-hf/piqa",
    }
}

MODEL_CHOICES = ['7b']
DATA_CHOICES = ['piqa']
CONFIG = {
    'num_layer': 32,
    'd_model': 4096,
    'intermediate': 11008,
    'samples_to_learn': 500000,
    'samples_per_file': 50000
}

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
    
def sigmoid(x_elem):
  return 1/(1 + np.exp(-x_elem))

def silu(x_elem, theda = 1.0):
    return x_elem * sigmoid(theda *x_elem)

def load_recorded_act(path: str, layer_idx: str) -> torch.Tensor:
    return torch.load(os.path.join(path, f"{layer_idx}_mlp.act_fn_mlp_activation_input.pt"))

def get_data(samples_to_learn, range_min=-0.5, range_max=0.5):
    activation_recorded_res_path = '/root/autodl-tmp/mlp_activation/Llama-2-7b-chat-hf/piqa'
    query = load_recorded_act(activation_recorded_res_path, 0).reshape(-1, 1)[:samples_to_learn].float()
    silu = nn.SiLU()
    label = silu(query)
    return query, label
    
    # query = []
    # label = []
    # for _ in range(samples_to_learn):
    #     x = random.uniform(range_min, range_max)
    #     y = silu(x)
    #     query.append(float(x))
    #     label.append(float(y))
    # return torch.tensor(query).unsqueeze(-1), torch.tensor(label).unsqueeze(-1)

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
    parser = argparse.ArgumentParser(description="Silu Predictor")
    parser.add_argument("--model", type=str, default="7b", choices = MODEL_CHOICES)
    parser.add_argument("--dataset", type=str, default="piqa", choices = DATA_CHOICES)
    parser.add_argument(
        "--L",
        type=int,
        default=0,
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
        default=640,
        help="batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
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

    query, labels = get_data(CONFIG['samples_to_learn'])

    train_loader, test_loader = create_dataset(query, labels, args)

    query_layer = torch.nn.Sequential(
        torch.nn.Linear(1, 1000, bias=None, dtype=torch.float32),
        torch.nn.Linear(1000, 1, bias=None, dtype=torch.float32)
    )

    print("Start Training")
    best_model, eval_result = train(
        query_layer,  train_loader, test_loader, args, device, verbal=True
    )

    dir = f"/root/autodl-tmp/predictors/silu-predictor"
    file = f"model.pt"
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save(best_model, os.path.join(dir, file))

if __name__ == "__main__":
    main()