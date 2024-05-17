import torch
import copy
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter

def eval_print(validation_results):
    result = ""
    for metric_name, metirc_val in validation_results.items():
        result = f"{result}{metric_name}: {metirc_val:.4f} "
    return result


def generate_label(y):
    # positive
    one_hot = (y > 0).to(y.dtype)
    return one_hot


def evaluate(model, device, loader, args, smalltest=False, loss_fn=nn.MSELoss()):
    model.eval()

    eval = {
        "Loss": []
    }
    with torch.no_grad():
        for batch_idx, batch in enumerate((loader)):
            x, y = batch
            y = y.to(device).nan_to_num(nan=0.0, posinf=100.0, neginf=-100.0)
            assert y.isnan().sum() == 0 and y.isinf().sum() == 0
            y_pred = model(x.float().to(device))

            eval["Loss"] += [
                loss_fn(y, y_pred).cpu()
            ]
            if batch_idx >= 100 and smalltest:
                break

    for k, v in eval.items():
        eval[k] = np.array(v).mean()

    return eval


def train(model, train_loader, valid_loader, args, device, verbal=True):
    writer = SummaryWriter()
    
    num_val = 0
    early_stop_waiting = 5
    val_inter = len(train_loader) // (num_val + 1) + 1
    num_print = 0
    print_inter = len(train_loader) // (num_print + 1) + 1
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.01)
    loss_fn = nn.MSELoss()

    eval_results = evaluate(model, device, valid_loader, args, smalltest=True)
    if verbal:
        print(f"[Start] {eval_print(eval_results)}")

    best_model = copy.deepcopy(model.state_dict())
    # base_acc = eval_results["Recall"]
    best_eval = eval_results
    no_improve = 0
    for e in range(args.epochs):
        model.train()
        for _, batch in enumerate(tqdm(train_loader)):
            x, y = batch
            optimizer.zero_grad()

            # not sure if this is correct
            y = y.to(device).nan_to_num(nan=0.0, posinf=100.0, neginf=-100.0)
            assert y.isnan().sum() == 0 and y.isinf().sum() == 0
            y_pred = model(x.float().to(device))
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

        train_eval_results = evaluate(model, device, train_loader, args, smalltest=True)
        epoch_eval_results = evaluate(
            model, device, valid_loader, args, smalltest=False
        )
        if verbal:
            print(f"[Epoch {e+1}] [Train] {eval_print(train_eval_results)}")
            print(f"[Epoch {e+1}] [Valid] {eval_print(epoch_eval_results)}\n")
        writer.add_scalar("Loss/train", train_eval_results["Loss"], e)
        writer.add_scalar("Loss/eval", epoch_eval_results["Loss"], e)
        writer.flush()

        # if epoch_eval_results["Recall"] > base_acc:
        #     base_acc = epoch_eval_results["Recall"]
        #     best_eval = epoch_eval_results
        #     model.cpu()
        #     best_model = copy.deepcopy(model.state_dict())
        #     model.to(device)
        #     no_improve = 0
        # else:
        #     no_improve += 1

        # if no_improve >= early_stop_waiting or base_acc > 0.99:
        #     break
    writer.close()
    return best_model, best_eval
