import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoModel


class GLUE_Dataset(Dataset):
    def __init__(self, data_df, data_cols, eval_col, tokenizer) -> None:
        super().__init__()
        self.data_df = data_df
        self.data_cols = data_cols
        self.eval_col = eval_col
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        if len(self.data_cols) == 1:
            tok_input = self.data_df[self.data_cols[0]].iloc[index]
            tok_output = self.tokenizer(
                text=tok_input,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
        else:
            tok_input = (
                self.data_df[self.data_cols[0]].iloc[index],
                self.data_df[self.data_cols[1]].iloc[index],
            )
            tok_output = self.tokenizer(
                *tok_input,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

        return_dict = {
            "input_ids": tok_output["input_ids"],
            "attention_mask": tok_output["attention_mask"],
            "token_type_ids": tok_output["token_type_ids"],
            "labels": self.data_df[self.eval_col].iloc[index],
        }
        return return_dict


def get_data_df(conf, train=False):
    data_dir = conf["data_dir"]
    df_path = conf["task_folder"]
    if train:
        df_path = f"{data_dir}/{df_path}/train.tsv"
    else:
        df_path = f"{data_dir}/{df_path}/dev.tsv"
    h = 0
    if conf["task_folder"] == "CoLA":
        h = None

    df = pd.read_csv(df_path, sep="\t", header=h)
    if conf["task_folder"] == "RTE":
        df["bin_label"] = (df["label"] != "entailment") * 1.0
    return df


def load_tokenizer_and_model(conf):
    model_name = conf["model"]
    tok_name = conf["tokenizer"]
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, output_attentions=True
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    return tokenizer, model
