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
    'samples_per_file': 500000
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
    
def load_recorded_input_data(path: str, layer_idx: str) -> torch.Tensor:
    return torch.load(os.path.join(path, f"{layer_idx}_mlp_mlp_input_res.pt"))

def load_recorded_label_data(path: str, layer_idx: str) -> torch.Tensor:
    return torch.load(os.path.join(path, f"{layer_idx}_mlp_input_x_hat_labels.pt"))

def get_data(samples_to_learn, layer_idx=0):
    recorded_inputs_path = '/root/autodl-tmp/mlp_activation/Llama-2-7b-chat-hf/piqa'
    query = load_recorded_input_data(recorded_inputs_path, layer_idx).float()
    label = load_recorded_input_data(recorded_inputs_path, layer_idx).float()
    return query, label
    
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
    parser = argparse.ArgumentParser(description="X hat Predictor")
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
        default=256,
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

    query, labels = get_data(CONFIG['samples_to_learn'], args.L)

    train_loader, test_loader = create_dataset(query, labels, args)

    query_layer = torch.nn.Sequential(
        torch.nn.Linear(CONFIG['d_model'], args.D, bias=None, dtype=torch.float32),
        torch.nn.SiLU(),
        torch.nn.Linear(args.D, CONFIG['d_model'], bias=None, dtype=torch.float32),
    )

    print("Start Training")
    best_model, eval_result = train(
        query_layer,  train_loader, test_loader, args, device, verbal=True
    )

    dir = f"/root/autodl-tmp/predictors/x-hat-predictor"
    file = f"model.pt"
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save(best_model, os.path.join(dir, file))

if __name__ == "__main__":
    main()