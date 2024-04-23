from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def calc_accuracy(preds, ground_truth):
    acc = accuracy_score(ground_truth, preds)
    return acc


def calc_matthew_correlation(preds, ground_truth):
    mcc = matthews_corrcoef(ground_truth, preds)
    return mcc


def calc_f1_score(preds, ground_truth):
    f1 = f1_score(ground_truth, preds)
    return f1


def eval_model(model, dataset, batch_size=32):
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    gt = []
    for x, y in eval_dataloader:
        batch_preds = model(x).argmax(dim=1)
        preds.append(batch_preds.to_list())
        gt.append(y.to_list())
    final_metrics = {
        "acc": calc_accuracy(preds, gt),
        "f1": calc_f1_score(preds, gt),
        "mcc": calc_matthew_correlation(preds, gt),
    }
    return final_metrics
