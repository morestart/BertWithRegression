from __future__ import annotations

import platform
from typing import List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
# from torchnlp.utils import collate_tensors
from transformers import BertTokenizer

from bert.encoders.label_encoder import LabelEncoder


def is_windows() -> bool:
    return platform.system().lower() == "windows"


class RegressionDataset(Dataset):
    def __init__(
            self,
            data: str | np.ndarray,
            drop_columns: List[str],
            max_length: int,
            tokenizer: BertTokenizer,
    ):
        super().__init__()

        self.tokenizer = tokenizer

        if isinstance(data, str):
            self.df = pd.read_csv(data)
            for item in drop_columns:
                data = data.drop(columns=item)
            self.df = data.values
        elif isinstance(data, pd.DataFrame):
            for item in drop_columns:
                data = data.drop(columns=item)
            self.df = data.values
        else:
            logger.error('不支持的df类型')
            exit(-1)

        self.max_length = max_length

    def __getitem__(self, item):
        text, label = self.df[item]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length',
                                  truncation=True)

        return {'input_ids': encoding['input_ids'][0],
                'attention_mask': encoding['attention_mask'][0]}, float(label)

    def __len__(self):
        return len(self.df)


class RegressionDataModule(pl.LightningDataModule):
    def __init__(
            self,
            csv_data_path: str,
            drop_columns: List[str],
            max_length: int,
            tokenizer: BertTokenizer,
            batch_size: int = 32,
            val_split: float = 0.2,
            test_split: float = 0.1,
            num_workers: int = 4,
            label_set: List[str] | None = None
    ):
        super().__init__()

        if label_set is None:
            label_set = ['E', 'N', 'H', 'T', 'F', 'I', 'R', 'S', 'K', 'Q', 'L', 'Y', 'G', 'A', 'V', 'C', 'W',
                         'D', 'M', 'P']
        self.csv_data_path = csv_data_path
        self.drop_columns = drop_columns
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = 0 if is_windows() else num_workers

        self.label_encoder = LabelEncoder(
            label_set, reserved_labels=[]
        )
        # print(self.label_encoder.vocab_size)
        self.label_encoder.unknown_index = None

        self.setup()

    # def prepare_sample(self, sample: list) -> (dict, dict):
    #     """
    #     Function that prepares a sample to input the model.
    #     :param prepare_target:
    #     :param sample: list of dictionaries.
    #
    #     Returns:
    #         - dictionary with the expected model inputs.
    #         - dictionary with the expected target labels.
    #     """
    #     sample = collate_tensors(sample)
    #
    #     sample_list = [x[0] for x in sample]
    #     target_list = [float(x[1]) for x in sample]
    #
    #     inputs = self.tokenizer.batch_encode_plus(sample_list,
    #                                               add_special_tokens=True,
    #                                               padding=True,
    #                                               truncation=True,
    #                                               max_length=self.max_length)
    #     # Prepare target:
    #     try:
    #         # targets = {"labels": self.label_encoder.batch_encode(sample["label"])}
    #         return inputs, sample[1]
    #     except RuntimeError:
    #         # print(sample["label"])
    #         raise Exception("Label encoder found an unknown label.")

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        df = pd.read_csv(self.csv_data_path)

        train_df, test_df = train_test_split(df, test_size=self.test_split, random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=self.val_split, random_state=42)

        self.train_dataset = RegressionDataset(train_df, self.drop_columns, self.max_length, self.tokenizer)
        self.val_dataset = RegressionDataset(val_df, self.drop_columns, self.max_length, self.tokenizer)
        self.test_dataset = RegressionDataset(test_df, self.drop_columns, self.max_length, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


if __name__ == '__main__':
    dl = RegressionDataModule(
        csv_data_path='/home/jp/Documents/Bert/data/result-final.csv',
        drop_columns=['uniprot_id', 'all_mutation_str'],
        max_length=2477,
        tokenizer=BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd"),
    )
    for i in dl.train_dataloader():
        print()
