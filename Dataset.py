#!/usr/bin/env python

# @Time    : 2024/9/4 15:21
# @Author  : Yao Xuan
# @Email   : xuany0512@163.com
# @File    : Dataset.py

from torch.utils.data import Dataset
import pandas as pd

class HLAPEPDataset(Dataset):
    def __init__(self,file_path):
        self.data = pd.read_csv(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        hla_seq = self.data['mhc_seq'][idx]
        pep_seq = self.data['pep'][idx]
        label = self.data['label'][idx]
        return hla_seq, pep_seq, label

    