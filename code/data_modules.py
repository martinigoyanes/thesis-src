import os
from typing import Optional
from datasets import ParaDetoxDataset
import pytorch_lightning as pl
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

class ParaDetoxDM(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.tokenizer_name_or_path = args.tokenizer_name_or_path
        self.max_seq_len = args.max_seq_len
        self.batch_size = args.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, use_fast=True)
        self.datasets = {}
    
    def prepare_data(self):
        AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, use_fast=True)

    def setup(self, stage: Optional[str]):
        if stage == "predict":
            self.datasets['predict'] = ParaDetoxDataset(split='predict', tokenizer=self.tokenizer)

    def predict_dataloader(self):
        return DataLoader(self.datasets['predict'], batch_size = self.batch_size, num_workers=os.cpu_count(), shuffle=False)
    