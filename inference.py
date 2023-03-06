import torch
from typing import Optional
import os
import pytorch_lightning as pl
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler
from transformers import BartForConditionalGeneration, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

class ParaDetoxDataset(Dataset):

  def __init__(self, split, tokenizer, max_seq_len=None):
    self.split = split
    self.tokenizer = tokenizer
    self.max_seq_len = max_seq_len if max_seq_len else None
    self.setup()

  def setup(self):
    if self.split == "predict":
        data_dir = '/Midgard/home/martinig/thesis-src/data/paradetox/test_toxic.txt'
        with open(data_dir, 'r') as f:
            texts = [line.strip() for line in f.readlines()]
        self.data = self.tokenizer(texts, return_tensors='pt', padding=True)
    
  def __len__(self):
    return len(self.data['input_ids'])

  def __getitem__(self, index):
    return {'input_ids': self.data['input_ids'][index], 'attention_mask': self.data['attention_mask'][index]}

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

class BARTdetox(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.model_name_or_path = args.model_name_or_path
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name_or_path)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        loss, outputs = self(**batch)
        self.log("train loss ", loss, prog_bar = True, logger=True)
        return {'loss':loss}

    def predict_step(self, batch, batch_idx):
        preds = self.generate(
            inputs=batch["input_ids"],
            num_return_sequences=1,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=3.0,
            num_beams=10
        )
        return preds
    
def save_preds(preds, out_dir, tokenizer):
    path = f"{out_dir}/preds.txt"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    for pred_ids in preds:
        prediction_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        with open(path, 'a') as f: f.write('\n'.join(prediction_texts)+'\n')


if __name__ == "__main__":
    from argparse import  ArgumentParser

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument("--model_name_or_path", type=str, default='s-nlp/bart-base-detox')
    parser.add_argument("--tokenizer_name_or_path", type=str, default='facebook/bart-base')
    parser.add_argument("--batch_size", type=int, default=32) # I made this up
    parser.add_argument("--max_seq_len", type=int, default=None) # Dont use for inference
    parser.add_argument("--out_dir", type=str, default='.')
    args = parser.parse_args()

    pl.seed_everything(44)

    bart_detox = BARTdetox(args)
    data_module = ParaDetoxDM(args)
    # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # trainer = pl.Trainer.from_argparse_args(args, fast_dev_run=True, deterministic=True, profiler='advanced')
    trainer = pl.Trainer.from_argparse_args(args, accelerator="gpu", devices=1, default_root_dir=args.out_dir)
    preds = trainer.predict(model=bart_detox, datamodule=data_module)
    save_preds(preds=preds, out_dir=args.out_dir, tokenizer=data_module.tokenizer)