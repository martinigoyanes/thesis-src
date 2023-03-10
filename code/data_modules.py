import os
from typing import Optional
from datasets import ParaDetoxDataset
from datasets import YelpDataset
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
    
class YelpDM(pl.LightningDataModule):

    def __init__(self, tokenizer_name_or_path: str, max_seq_len: int, batch_size: int, preprocess_kind: str = None, **kwargs):
        super().__init__()
        assert preprocess_kind
        self.preprocess_kind = preprocess_kind
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.special_tokens = ['<POS>', '<NEG>','<CON_START>','<START>','<END>', '<PAD>']
        self.datasets = {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path, 
            use_fast=True, 
            additional_special_tokens=self.special_tokens,
            pad_token='<PAD>',
            eos_token='<END>',
            bos_token='<START>'
        )
    
    @staticmethod
    def add_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("YelpDM")
        parser.add_argument("--tokenizer_name_or_path", type=str, default='openai-gpt')
        parser.add_argument("--batch_size", type=int, default=8) # I made this up
        parser.add_argument("--max_seq_len", type=int, default=110) # From paper
        return parent_parser

    def prepare_data(self):
        AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, use_fast=True, additional_special_tokens=self.special_tokens)
        YelpDataset(split='train', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind, max_seq_len=self.max_seq_len, prepare_data=True)
        YelpDataset(split='dev', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind, max_seq_len=self.max_seq_len, prepare_data=True)
        # YelpDataset(split='test', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind, prepare_data=True)

    def setup(self, stage: Optional[str]):
        if stage == "fit":
            self.datasets['train'] = YelpDataset(split='train', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind)
            self.datasets['dev'] = YelpDataset(split='dev', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind)

    def predict_dataloader(self):
        return DataLoader(self.datasets['predict'], batch_size = self.batch_size, num_workers=os.cpu_count(), shuffle=False)
    
    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size = self.batch_size, num_workers=os.cpu_count(), shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.datasets['dev'], batch_size = self.batch_size, num_workers=os.cpu_count(), shuffle=False)
    