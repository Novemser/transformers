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

def evaluate(model, device, loader, args, smalltest=False, loss_fn=nn.MSELoss(), classification=False):
    model.eval()

    eval = {
        "Loss": []
    } if not classification else {
        "Loss": [],
        "Loss Weight": [],
        "Recall": [],
        "Classifier Sparsity": [],
        "True Sparsity": [],
    }
    with torch.no_grad():
        for batch_idx, batch in enumerate((loader)):
            x, y = batch
            y_pred = model(x.float().to(device))
            
            if classification:
                y_pred = y_pred.sigmoid()
                preds = y_pred >= 0.5
                dif = y.int() - preds.int()
                miss = dif > 0.0  # classifier didn't activated target neuron
                weight = (y.sum() / y.numel()) + 0.005
                loss_weight = y * (1 - weight) + weight
                eval["Loss Weight"] += [weight.item()]
                eval["Recall"] += [
                    ((y.sum(dim=1).float() - miss.sum(dim=1).float()).mean().item())
                ]
                eval["True Sparsity"] += [y.sum(dim=1).float().mean().item()]
                eval["Classifier Sparsity"] += [preds.sum(dim=1).float().mean().item()]

            eval["Loss"] += [
                loss_fn(y, y_pred).cpu() if not classification else torch.nn.functional.binary_cross_entropy(
                    y_pred, y, weight=loss_weight
                ).item()
            ]
            if batch_idx >= 100 and smalltest:
                break

    for k, v in eval.items():
        eval[k] = np.array(v).mean()

    if classification:
        eval["Recall"] = eval["Recall"] / eval["True Sparsity"]
    return eval


def train(model, train_loader, valid_loader, args, device, verbal=True, classification=False):
    writer = SummaryWriter()
    
    early_stop_waiting = 5
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.01)
    loss_fn = nn.MSELoss()

    eval_results = evaluate(model, device, valid_loader, args, smalltest=True, classification=classification)
    if verbal:
        print(f"[Start] {eval_print(eval_results)}")

    best_model = copy.deepcopy(model.state_dict())
    base_loss = eval_results["Loss"]
    if classification:
        base_acc = eval_results["Recall"]
    best_eval = eval_results
    no_improve = 0
    for e in range(args.epochs):
        model.train()
        for _, batch in enumerate(tqdm(train_loader)):
            x, y = batch
            optimizer.zero_grad()

            y_pred = model(x.float().to(device))
            if classification:
                y_pred = y_pred.sigmoid()
                weight = (y.sum() / y.numel()) + 0.005
                loss_weight = y * (1 - weight) + weight

            loss = loss_fn(y_pred, y) if not classification else\
                torch.nn.functional.binary_cross_entropy(y_pred, y, weight=loss_weight)
            loss.backward()
            optimizer.step()

        train_eval_results = evaluate(model, device, train_loader, args, smalltest=True, classification=classification)
        epoch_eval_results = evaluate(
            model, device, valid_loader, args, smalltest=False, classification=classification
        )
        if verbal:
            print(f"[Epoch {e+1}] [Train] {eval_print(train_eval_results)}")
            print(f"[Epoch {e+1}] [Valid] {eval_print(epoch_eval_results)}\n")
        writer.add_scalar("Loss/train", train_eval_results["Loss"], e)
        writer.add_scalar("Loss/eval", epoch_eval_results["Loss"], e)
        writer.flush()

        if classification and epoch_eval_results["Recall"] > base_acc:
            base_acc = epoch_eval_results["Recall"]
            best_eval = epoch_eval_results
            model.cpu()
            best_model = copy.deepcopy(model.state_dict())
            model.to(device)
            no_improve = 0
        elif not classification and epoch_eval_results["Loss"] < base_loss:
            base_loss = epoch_eval_results["Loss"]
            best_eval = epoch_eval_results
            model.cpu()
            best_model = copy.deepcopy(model.state_dict())
            model.to(device)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= early_stop_waiting or \
            (classification and base_acc >= 0.999) or\
            (not classification and base_loss <= 1e-6):
            break
    writer.close()
    return best_model, best_eval
