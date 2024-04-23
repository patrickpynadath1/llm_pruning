import pandas as pd
from torch.utils.data import Dataset, DataLoader

class GLUE_Dataset(Dataset):
    def __init__(self, 
                 data_df,
                 data_cols, 
                 eval_col) -> None:
        super().__init__()
        self.data_df = data_df
        self.data_cols = data_cols 
        self.eval_col = eval_col 

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, index):
        if len(self.data_cols) == 1: 
            return_dict = (
                self.data_df.iloc[index][self.data_cols[0]],
                self.data_df.iloc[index][self.eval_col]
            )
        else: 
            return_dict = (
                self.data_df.iloc[index][self.data_cols[0]],
                self.data_df.iloc[index][self.data_cols[1]],
                self.data_df.iloc[index][self.eval_col]
            )
        return return_dict
    

def get_data_df(data_dir, task, train=False):
    if train: 
        task = f"{data_dir}/{task}/train.tsv"
    else: 
        task = f"{data_dir}/{task}/test.tsv"
    h=None
    if task == 'CoLA':
        h = 0

    df = pd.read_csv(task, sep='\t', header=h)
    return df


def construct_dataset(data_df): 
    return 




