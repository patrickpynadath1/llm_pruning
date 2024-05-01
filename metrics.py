from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm import tqdm


def calc_accuracy(preds, ground_truth):
    acc = accuracy_score(ground_truth, preds)
    return acc


def calc_matthew_correlation(preds, ground_truth):
    mcc = matthews_corrcoef(ground_truth, preds)
    return mcc


def calc_f1_score(preds, ground_truth):
    f1 = f1_score(ground_truth, preds)
    return f1


def eval_model(model, dataset):
    preds = []
    gt = []
    for i in tqdm(range(len(dataset))):
        try: 
            x = dataset[i]
            y = x["labels"]
            out = model(
                input_ids=x["input_ids"],
                attention_mask=x["attention_mask"],
                token_type_ids=x["token_type_ids"],
            )
            logits = out.logits
            preds.append(logits.argmax(dim=-1).tolist())
            gt.append(y.tolist())
        except: 
            print("Error in eval_model")
            continue
    final_metrics = {
        "acc": calc_accuracy(preds, gt),
        "f1": calc_f1_score(preds, gt),
        "mcc": calc_matthew_correlation(preds, gt),
        "model_size": model.num_parameters(),
    }
    return final_metrics
