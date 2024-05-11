import torch
import numpy as np
import argparse
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from trainer_mlp_scaler import train

DATA = {
    "7b": {
        "piqa": "/root/autodl-tmp/mlp_activation/Llama-2-7b-chat-hf/piqa",
    }
}

MODEL_CHOICES = ['7b']
DATA_CHOICES = ['piqa']
CONFIG = {
    '7b':{
        'num_layer': 95,
        'd_model': 4096,
        'intermediate': 11008,
        'layers': 32,
        'samples_to_learn': 100,
        'samples_per_file': 100
    }
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

def get_data(args, layer_num):
    path = f"{DATA[args.model][args.dataset]}/{layer_num}_mlp.act_fn_mlp_activation_res.pt"
    print(f"Reading query from {path}")
    query = torch.load(path)[:CONFIG[args.model]['samples_to_learn']]

    for part_index in range(CONFIG[args.model]['samples_to_learn'] // CONFIG[args.model]['samples_per_file']):
        path = f"{DATA[args.model][args.dataset]}/{layer_num}_mlp.act_fn_mlp_activation_res_labels_part_{part_index}.pt"
        print(f"Reading MLP label {part_index} from {path}")
        label = torch.load(path)
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
    parser = argparse.ArgumentParser(description="PyTorch LLaMa 7B-hf Activation Scaler Predictor")
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
        default=1,
        help="low rank dimension",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="learning rate",
    )
    args = parser.parse_args()

    print(args)
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("=" * 40, "Chosen Layer", args.L, "=" * 40)

    query, labels = get_data(args, args.L)

    train_loader, test_loader = create_dataset(query, labels, args)

    query_layer = torch.nn.Sequential(
        torch.nn.Linear(CONFIG[args.model]['intermediate'], args.D, bias=None), # intermediate * D
        torch.nn.Linear(args.D, CONFIG[args.model]['d_model'] ** 2, bias=None), # D * d_model^2
    )

    print("Start Training")
    best_model, eval_result = train(
        query_layer,  train_loader, test_loader, args, device, verbal=True
    )

    path = f"/root/autodl-tmp/predictors/llama2-{args.model}-scaler-predictor/{args.dataset}_layer{args.L}.pt"
    torch.save(best_model, path)

if __name__ == "__main__":
    main()