import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoModel
import numpy as np
import torch


class GLUE_Dataset(Dataset):
    def __init__(self, 
                 data_df, 
                 data_cols, 
                 eval_col, 
                 tokenizer, 
                 rand_masking=0.5,
                 rand_gibberish = 0.5,
                 mode="normal") -> None:
        super().__init__()
        self.data_df = data_df
        self.data_cols = data_cols
        self.eval_col = eval_col
        self.tokenizer = tokenizer
        self.rand_masking = rand_masking
        self.rand_gibberish = rand_gibberish
        self.mode = mode

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
        if self.rand_masking > 0 and np.random.random() < self.rand_masking: 
            tok_output["input_ids"] = self.random_mask(tok_output["input_ids"])
        if self.rand_gibberish > 0 and np.random.random() < self.rand_gibberish:
            tok_output["input_ids"] = self.random_gibberish(tok_output["input_ids"])


        return_dict = {
            "input_ids": tok_output["input_ids"],
            "attention_mask": tok_output["attention_mask"],
            "token_type_ids": tok_output["token_type_ids"],
            "labels": torch.tensor(self.data_df[self.eval_col].iloc[index].values),
        }
        return return_dict
    

    def random_mask(self, tokens):
        sentence_length = get_sentence_length(tokens)
        tokens_to_return = tokens.clone()
        percentage = self.rand_masking
        indices = list(np.random.choice([i for i in range(1, sentence_length-1)], int(sentence_length * percentage), replace=False))
        tokens_to_return[0, indices] =  103
        return tokens_to_return

    def random_gibberish(self, tokens):
        sentence_length = get_sentence_length(tokens)
        tokens_to_return = tokens.clone()
        percentage = self.rand_gibberish
        indices = list(np.random.choice([i for i in range(1, sentence_length-1)], int(sentence_length * percentage), replace=False))
        rand_tokens = list(np.random.randint(0, 30522, len(indices)))
        tokens_to_return[0, indices] = torch.tensor(rand_tokens)
        return tokens_to_return


def get_sentence_length(tokens):
    sentence_length  = (tokens == 0).nonzero(as_tuple=True)[1][0]
    return sentence_length


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
