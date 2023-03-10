
import os
import torch
from torch.utils.data import Dataset
import logging
logger = logging.getLogger(__name__)

class ParaDetoxDataset(Dataset):

  def __init__(self, split, tokenizer, max_seq_len=None):
    super().__init__()
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

class YelpDataset(Dataset):

  def __init__(self, split, tokenizer, max_seq_len=None, preprocess_kind: str = None, prepare_data: bool = False):
    super().__init__()
    assert preprocess_kind
    self.preprocess_kind = preprocess_kind
    self.split = split
    self.tokenizer = tokenizer
    self.max_seq_len = max_seq_len if max_seq_len else None
    self.data_dir = f'/Midgard/home/martinig/thesis-src/data/yelp/{self.preprocess_kind}'
    self.cache_dir = f'{self.data_dir}/.cache'
    self.prepare_data = prepare_data
    self.setup()

  def setup(self):
    def _get_index(tensor, value):
      return (tensor == value).nonzero()[0]

    if self.split == "train":
        cache_file = f"{self.cache_dir}/sentiment.train.tokenized.pt"
        data_file = f"{self.data_dir}/sentiment.train"

    if not os.path.exists(cache_file): 
      os.makedirs(self.cache_dir, exist_ok=True)

      with open(data_file, 'r') as f:
          texts = [line.strip() for line in f.readlines()]

      logger.info(f"Tokenizing data from {data_file}...")
      self.data = self.tokenizer(texts, return_tensors='pt', padding=True)

      logger.info(f"Masking labels from {data_file}...")
      size = self.data.input_ids.size()
      labels = torch.full(size=size, fill_value=-100)
      for i, ids in enumerate(self.data.input_ids):
        idx_start = _get_index(ids, self.tokenizer.bos_token_id) 
        idx_end = _get_index(ids, self.tokenizer.eos_token_id) 
        labels[i, idx_start:idx_end] = ids[idx_start:idx_end]
      
      self.data['labels'] = labels         
      logger.info(f"Saving tokenization to cache [{cache_file}] ...")
      with open(cache_file, 'wb') as f:
        torch.save(obj=self.data, f=f)

    if not self.prepare_data:
      logger.info(f"Loading tokenization from cache [{cache_file}] ...")
      self.data = torch.load(cache_file)
    
  def __len__(self):
    return len(self.data['input_ids'])

  def __getitem__(self, index):
    return {
      'input_ids': self.data['input_ids'][index], 
      'attention_mask': self.data['attention_mask'][index],
      'labels': self.data['labels'][index]
    }