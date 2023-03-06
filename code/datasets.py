
from torch.utils.data import Dataset

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
