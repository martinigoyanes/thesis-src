import os
from typing import Optional
from datasets import ParaDetoxDataset
from datasets import YelpDataset
from datasets import OriginalYelpDataset2
from datasets import OriginalYelpDataset
from datasets import OriginalJigsawDataset
import pytorch_lightning as pl
from transformers import AutoTokenizer
from pytorch_pretrained_bert import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIAdam, cached_path
from torch.utils.data import DataLoader

class ParaDetoxDM(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
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
    '''
        Datamodule to train GPT from transformers on Yelp data processed by BERT
    '''

    def __init__(self, tokenizer_name_or_path: str, max_seq_len: int, batch_size: int, preprocess_kind: str = None, **kwargs):
        super().__init__()
        assert preprocess_kind
        self.save_hyperparameters()
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
        parser.add_argument("--batch_size", type=int, default=32) # I made this up
        parser.add_argument("--max_seq_len", type=int, default=110) # From paper
        return parent_parser

    def prepare_data(self):
        AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, use_fast=True, additional_special_tokens=self.special_tokens)
        YelpDataset(split='train', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind, max_seq_len=self.max_seq_len, prepare_data=True)
        YelpDataset(split='dev', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind, max_seq_len=self.max_seq_len, prepare_data=True)
        YelpDataset(split='test', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind, prepare_data=True)

    def setup(self, stage: Optional[str]):
        if stage == "fit":
            self.datasets['train'] = YelpDataset(split='train', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind)
            self.datasets['dev'] = YelpDataset(split='dev', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind)
        if stage == "predict":
            self.datasets['test'] = YelpDataset(split='test', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind)
    

    def predict_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size = 1, num_workers=os.cpu_count(), shuffle=False)
    
    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size = self.batch_size, num_workers=os.cpu_count(), shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.datasets['dev'], batch_size = self.batch_size, num_workers=os.cpu_count(), shuffle=False)
    

class OriginalJigsawDM(pl.LightningDataModule):
    '''
        Datamodule to train GPT from pytorch_pretrained_bert on Jigsaw data processed by BERT
    '''

    def __init__(self, tokenizer_name_or_path: str, max_seq_len: int, batch_size: int, preprocess_kind: str = None, **kwargs):
        super().__init__()
        assert preprocess_kind
        self.save_hyperparameters()
        self.preprocess_kind = preprocess_kind
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.special_tokens = ['<NEU>', '<TOX>','<CON_START>','<START>','<END>', '<PAD>']
        self.datasets = {}
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained(
            self.tokenizer_name_or_path, 
            special_tokens=self.special_tokens,
        )
    
    @staticmethod
    def add_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("OriginalJigsawDM")
        parser.add_argument("--tokenizer_name_or_path", type=str, default='openai-gpt')
        parser.add_argument("--batch_size", type=int, default=32) # I made this up
        parser.add_argument("--max_seq_len", type=int, default=128) # I made this one up
        return parent_parser

    def prepare_data(self):
        OpenAIGPTTokenizer.from_pretrained(
            self.tokenizer_name_or_path, 
            special_tokens=self.special_tokens,
        )
        OriginalJigsawDataset(split='train', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind, max_seq_len=self.max_seq_len, prepare_data=True)
        OriginalJigsawDataset(split='dev', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind, max_seq_len=self.max_seq_len, prepare_data=True)
        OriginalJigsawDataset(split='test', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind, prepare_data=True)

    def setup(self, stage: Optional[str]):
        if stage == "fit":
            self.datasets['train'] = OriginalJigsawDataset(split='train', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind)
            self.datasets['dev'] = OriginalJigsawDataset(split='dev', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind)
        if stage == "predict":
            self.datasets['test'] = OriginalJigsawDataset(split='test', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind)
    

    def predict_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size = self.batch_size, num_workers=os.cpu_count(), shuffle=False)
    
    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size = self.batch_size, num_workers=os.cpu_count(), shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.datasets['dev'], batch_size = self.batch_size, num_workers=os.cpu_count(), shuffle=False)

class OriginalYelpDM(pl.LightningDataModule):
    '''
        Datamodule to train GPT from pytorch_pretrained_bert on Yelp data processed by BERT
    '''

    def __init__(self, tokenizer_name_or_path: str, max_seq_len: int, batch_size: int, preprocess_kind: str = None, **kwargs):
        super().__init__()
        assert preprocess_kind
        self.save_hyperparameters()
        self.preprocess_kind = preprocess_kind
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.special_tokens = ['<POS>', '<NEG>','<CON_START>','<START>','<END>', '<PAD>']
        self.datasets = {}
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained(
            self.tokenizer_name_or_path, 
            special_tokens=self.special_tokens,
        )
    
    @staticmethod
    def add_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("OriginalYelpDM")
        parser.add_argument("--tokenizer_name_or_path", type=str, default='openai-gpt')
        parser.add_argument("--batch_size", type=int, default=32) # I made this up
        parser.add_argument("--max_seq_len", type=int, default=110) # From paper
        return parent_parser

    def prepare_data(self):
        OpenAIGPTTokenizer.from_pretrained(
            self.tokenizer_name_or_path, 
            special_tokens=self.special_tokens,
        )
        OriginalYelpDataset(split='train', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind, max_seq_len=self.max_seq_len, prepare_data=True)
        OriginalYelpDataset(split='dev', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind, max_seq_len=self.max_seq_len, prepare_data=True)
        # OriginalYelpDataset(split='test', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind, prepare_data=True)

    def setup(self, stage: Optional[str]):
        if stage == "fit":
            self.datasets['train'] = OriginalYelpDataset(split='train', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind)
            self.datasets['dev'] = OriginalYelpDataset(split='dev', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind)
        if stage == "predict":
            self.datasets['test'] = OriginalYelpDataset(split='test', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind)
    

    def predict_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size = self.batch_size, num_workers=os.cpu_count(), shuffle=False)
    
    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size = self.batch_size, num_workers=os.cpu_count(), shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.datasets['dev'], batch_size = self.batch_size, num_workers=os.cpu_count(), shuffle=False)
    
class YelpDM2(pl.LightningDataModule):
    '''
        Datamodule to train, eval and test RoBERTa fine-tuned on the original/raw Yelp data
    '''

    def __init__(self, tokenizer_name_or_path: str, max_seq_len: int, batch_size: int, preprocess_kind: str = None, **kwargs):
        super().__init__()
        assert preprocess_kind
        self.save_hyperparameters()
        self.preprocess_kind = preprocess_kind
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.datasets = {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path, 
            use_fast=True, 
        )
    
    @staticmethod
    def add_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("YelpDM")
        parser.add_argument("--tokenizer_name_or_path", type=str, default='roberta-base')
        parser.add_argument("--batch_size", type=int, default=32) # I made this up
        parser.add_argument("--max_seq_len", type=int, default=110) # From paper
        return parent_parser

    def prepare_data(self):
        AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, use_fast=True)
        OriginalYelpDataset2(split='train', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind, max_seq_len=self.max_seq_len, prepare_data=True)
        OriginalYelpDataset2(split='dev', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind, max_seq_len=self.max_seq_len, prepare_data=True)
        OriginalYelpDataset2(split='test', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind, prepare_data=True)

    def setup(self, stage: Optional[str]):
        if stage == "fit":
            self.datasets['train'] = OriginalYelpDataset2(split='train', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind)
            self.datasets['dev'] = OriginalYelpDataset2(split='dev', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind)
        if stage == "predict":
            self.datasets['test'] = OriginalYelpDataset2(split='test', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind)
        if stage == "test":
            self.datasets['test'] = OriginalYelpDataset2(split='test', tokenizer=self.tokenizer, preprocess_kind=self.preprocess_kind)
    

    def test_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size = self.batch_size, num_workers=os.cpu_count(), shuffle=False)
    
    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size = self.batch_size, num_workers=os.cpu_count(), shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.datasets['dev'], batch_size = self.batch_size, num_workers=os.cpu_count(), shuffle=False)

if __name__ == "__main__":

    dm = YelpDM2(
        tokenizer_name_or_path='roberta-base',
        max_seq_len=110,
        batch_size=32,
        preprocess_kind='original'
    )

    dm.setup(stage="fit")