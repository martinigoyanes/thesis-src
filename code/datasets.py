
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm, trange
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
    '''
        Dataset to train GPT from transformers on Yelp's data
    '''

    def __init__(self, split, tokenizer, max_seq_len=None, preprocess_kind: str = None, prepare_data: bool = False):
        super().__init__()
        assert preprocess_kind
        self.preprocess_kind = preprocess_kind
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len if max_seq_len else None
        self.data_dir = f'/Midgard/home/martinig/thesis-src/data/yelp/{self.preprocess_kind}'
        # self.data_dir = f'/home/martin/Documents/Education/Master/thesis/project/thesis-src/data/yelp/{self.preprocess_kind}'
        self.cache_dir = f'{self.data_dir}/.cache'
        self.prepare_data = prepare_data
        if self.split == "test": 
            self.tokenizer.padding_side = "left"
        self._setup()

    def _prepare_data(self, cache_file: str, data_file: str, create_labels: bool = False):

        with open(data_file, 'r') as f:
                texts = [line.strip() for line in f.readlines()]

        logger.info(f"Tokenizing {self.split} data from {data_file}...")
        self.data = self.tokenizer(texts, return_tensors='pt', padding=True)

        if create_labels:
            logger.info(f"Masking {self.split} labels from {data_file}...")
            self.data['labels'] = self._create_labels()         

        logger.info(f"Saving {self.split} tokenization to cache [{cache_file}] ...")
        with open(cache_file, 'wb') as f:
            torch.save(obj=self.data, f=f)
        
    def _create_labels(self):
        def _get_index(tensor, value):
            return (tensor == value).nonzero()[0]

        size = self.data.input_ids.size()
        labels = torch.full(size=size, fill_value=-100)

        for i, ids in enumerate(self.data.input_ids):
            idx_start = _get_index(ids, self.tokenizer.bos_token_id) 
            idx_end = _get_index(ids, self.tokenizer.eos_token_id) 
            labels[i, idx_start:idx_end] = ids[idx_start:idx_end]

        return labels
        


    def _setup(self):

        if self.split == "train":
                cache_file = f"{self.cache_dir}/sentiment.train.tokenized.pt"
                data_file = f"{self.data_dir}/sentiment.train"
        
        if self.split == 'dev':
                cache_file = f"{self.cache_dir}/sentiment.dev.tokenized.pt"
                data_file = f"{self.data_dir}/sentiment.dev"

        if self.split == 'test':
                cache_file = f"{self.cache_dir}/reference.0.tokenized.pt"
                data_file = f"{self.data_dir}/reference.0"

        os.makedirs(self.cache_dir, exist_ok=True)
        if not os.path.exists(cache_file): 
            self._prepare_data(cache_file=cache_file, data_file=data_file, create_labels=(self.split in ['train', 'dev']))

        if not self.prepare_data:
            logger.info(f"Loading {self.split} tokenization from cache [{cache_file}] ...")
            self.data = torch.load(cache_file)
        
    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, index):
        if self.split == "test":
            return {
                'input_ids': self.data['input_ids'][index], 
                'attention_mask': self.data['attention_mask'][index],
            }
        else:
            return {
                'input_ids': self.data['input_ids'][index], 
                'attention_mask': self.data['attention_mask'][index],
                'labels': self.data['labels'][index] 
            }

class OriginalJigsawDataset(Dataset):
    '''
        Dataset to train GPT from pytorch-pretrained-bert on Yelp data
    '''

    def __init__(self, split, tokenizer, max_seq_len=None, preprocess_kind: str = None, prepare_data: bool = False):
        super().__init__()
        assert preprocess_kind
        self.preprocess_kind = preprocess_kind
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len if max_seq_len else None
        self.data_dir = f'/Midgard/home/martinig/thesis-src/data/jigsaw/{self.preprocess_kind}'
        # self.data_dir = f'/home/martin/Documents/Education/Master/thesis/project/thesis-src/data/yelp/{self.preprocess_kind}'
        self.cache_dir = f'{self.data_dir}/.cache'
        self.prepare_data = prepare_data
        self._setup()

    def _prepare_data(self, cache_file: str, data_file: str, create_labels: bool = False):

        with open(data_file, 'r') as f:
                texts = [line.strip() for line in f.readlines()]

        logger.info(f"Tokenizing {self.split} data from {data_file}...")
        tokenized_dataset = texts
        for i, line in enumerate(tqdm(texts)):
                token = self.tokenizer.tokenize(line)
                tokenized_dataset[i] = self.tokenizer.convert_tokens_to_ids(token)
        self.data = self._preprocess_dataset(
             encoded_dataset=tokenized_dataset,
             input_length=self.max_seq_len,
             start_token_id=self.tokenizer.convert_tokens_to_ids(['<START>'])[0]
        )

        logger.info(f"Saving {self.split} tokenization to cache [{cache_file}] ...")
        with open(cache_file, 'wb') as f:
            torch.save(obj=self.data, f=f)
        
    def _preprocess_dataset(self,encoded_dataset, input_length, start_token_id):
        n_batch = len(encoded_dataset)
        input_ids = torch.zeros(size=(n_batch, input_length), dtype=torch.int64)
        lm_labels = torch.full(size=(n_batch, input_length), fill_value=-1, dtype=torch.int64)

        for i, tokens in enumerate(encoded_dataset):
            try:
                #tokens = tokens[:input_length]
                start_id_index = tokens.index(start_token_id)
                input_ids[i, :len(tokens)] = torch.Tensor(tokens)
                start_id_index = tokens.index(start_token_id)
                lm_labels[i, start_id_index : len(tokens)-1] = torch.Tensor(tokens[start_id_index + 1: len(tokens)])
                # LM loss calculate only for tokens after <START> token in the sentence
                #lm_labels[i, :len(tokens)-1] = tokens[1:]
            except ValueError:
                print("Index {} doesn't have start token".format(i))

        return {'input_ids': input_ids, 'lm_labels': lm_labels}
        
    def _setup(self):

        if self.split == "train":
            cache_file = f"{self.cache_dir}/train.tokenized.pt"
            data_file = f"{self.data_dir}/train"
        
        if self.split == 'dev':
            cache_file = f"{self.cache_dir}/dev.tokenized.pt"
            data_file = f"{self.data_dir}/dev"

        if self.split == 'test':
            data_file = f"{self.data_dir}/reference.toxic.in"
            with open(data_file, 'r') as f:
                    self.data = [line.strip() for line in f.readlines()]
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        if not os.path.exists(cache_file): 
            self._prepare_data(cache_file=cache_file, data_file=data_file, create_labels=(self.split in ['train', 'dev']))

        if not self.prepare_data:
            logger.info(f"Loading {self.split} tokenization from cache [{cache_file}] ...")
            self.data = torch.load(cache_file)
        
    def __len__(self):
        if self.split == "test": 
            return len(self.data)
        return len(self.data['input_ids'])

    def __getitem__(self, index):
        if self.split == 'test':
            return self.data[index]
        
        return {
            'input_ids': self.data['input_ids'][index], 
            'lm_labels': self.data['lm_labels'][index] 
        }


class OriginalYelpDataset(Dataset):
    '''
        Dataset to train GPT from pytorch-pretrained-bert on Yelp data
    '''

    def __init__(self, split, tokenizer, max_seq_len=None, preprocess_kind: str = None, prepare_data: bool = False):
        super().__init__()
        assert preprocess_kind
        self.preprocess_kind = preprocess_kind
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len if max_seq_len else None
        self.data_dir = f'/Midgard/home/martinig/thesis-src/data/yelp/{self.preprocess_kind}'
        # self.data_dir = f'/home/martin/Documents/Education/Master/thesis/project/thesis-src/data/yelp/{self.preprocess_kind}'
        self.cache_dir = f'{self.data_dir}/.cache'
        self.prepare_data = prepare_data
        self._setup()

    def _prepare_data(self, cache_file: str, data_file: str, create_labels: bool = False):

        with open(data_file, 'r') as f:
                texts = [line.strip() for line in f.readlines()]

        logger.info(f"Tokenizing {self.split} data from {data_file}...")
        tokenized_dataset = texts
        for i, line in enumerate(tqdm(texts)):
                token = self.tokenizer.tokenize(line)
                tokenized_dataset[i] = self.tokenizer.convert_tokens_to_ids(token)
        self.data = self._preprocess_dataset(
             encoded_dataset=tokenized_dataset,
             input_length=self.max_seq_len,
             start_token_id=self.tokenizer.convert_tokens_to_ids(['<START>'])[0]
        )

        logger.info(f"Saving {self.split} tokenization to cache [{cache_file}] ...")
        with open(cache_file, 'wb') as f:
            torch.save(obj=self.data, f=f)
        
    def _preprocess_dataset(self,encoded_dataset, input_length, start_token_id):
        n_batch = len(encoded_dataset)
        input_ids = torch.zeros(size=(n_batch, input_length), dtype=torch.int64)
        lm_labels = torch.full(size=(n_batch, input_length), fill_value=-1, dtype=torch.int64)

        for i, tokens in enumerate(encoded_dataset):
            try:
                #tokens = tokens[:input_length]
                start_id_index = tokens.index(start_token_id)
                input_ids[i, :len(tokens)] = torch.Tensor(tokens)
                start_id_index = tokens.index(start_token_id)
                lm_labels[i, start_id_index : len(tokens)-1] = torch.Tensor(tokens[start_id_index + 1: len(tokens)])
                # LM loss calculate only for tokens after <START> token in the sentence
                #lm_labels[i, :len(tokens)-1] = tokens[1:]
            except ValueError:
                print("Index {} doesn't have start token".format(i))

        return {'input_ids': input_ids, 'lm_labels': lm_labels}
        
    def _setup(self):

        if self.split == "train":
            cache_file = f"{self.cache_dir}/sentiment.train.tokenized.pt"
            data_file = f"{self.data_dir}/sentiment.train"
        
        if self.split == 'dev':
            cache_file = f"{self.cache_dir}/sentiment.dev.tokenized.pt"
            data_file = f"{self.data_dir}/sentiment.dev"

        if self.split == 'test':
            data_file = f"{self.data_dir}/reference.0.in"
            with open(data_file, 'r') as f:
                    self.data = [line.strip() for line in f.readlines()]
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        if not os.path.exists(cache_file): 
            self._prepare_data(cache_file=cache_file, data_file=data_file, create_labels=(self.split in ['train', 'dev']))

        if not self.prepare_data:
            logger.info(f"Loading {self.split} tokenization from cache [{cache_file}] ...")
            self.data = torch.load(cache_file)
        
    def __len__(self):
        if self.split == "test": 
            return len(self.data)
        return len(self.data['input_ids'])

    def __getitem__(self, index):
        if self.split == 'test':
            return self.data[index]
        
        return {
            'input_ids': self.data['input_ids'][index], 
            'lm_labels': self.data['lm_labels'][index] 
        }


class OriginalYelpDataset2(Dataset):
    '''
        Dataset to train, eval and test RoBERTa fine-tuned on the original/raw Yelp data
    '''

    def __init__(self, split, tokenizer, max_seq_len=None, preprocess_kind: str = None, prepare_data: bool = False):
        super().__init__()
        assert preprocess_kind
        self.preprocess_kind = preprocess_kind
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len if max_seq_len else None
        self.data_dir = f'/Midgard/home/martinig/thesis-src/data/yelp/{self.preprocess_kind}'
        # self.data_dir = f'/home/martin/Documents/Education/Master/thesis/project/thesis-src/data/yelp/{self.preprocess_kind}'
        self.cache_dir = f'{self.data_dir}/.cache'
        self.prepare_data = prepare_data
        self._setup()

    def _prepare_data(self, cache_file: str, data_file_0: str, data_file_1: str, create_labels: bool = False):

        with open(data_file_0, 'r') as f:
                texts_0 = [line.strip() for line in f.readlines()]
        with open(data_file_1, 'r') as f:
                texts_1 = [line.strip() for line in f.readlines()]

        logger.info(f"Tokenizing {self.split} data from {data_file_0} and {data_file_1} ...")
        texts = texts_0 + texts_1
        self.data = self.tokenizer(texts, return_tensors='pt', padding=True)

        if create_labels:
            logger.info(f"Masking {self.split} labels from {data_file_0} and {data_file_1} ...")
            num_samples_0, num_samples_1 = len(texts_0), len(texts_1)
            labels_0 = torch.zeros(size=(num_samples_0,), dtype= torch.long)
            labels_1 = torch.full(size=(num_samples_1,), fill_value=1, dtype=torch.long)
            self.data['labels'] = torch.cat((labels_0, labels_1))

        logger.info(f"Saving {self.split} tokenization to cache [{cache_file}] ...")
        with open(cache_file, 'wb') as f:
            torch.save(obj=self.data, f=f)
        
    def _setup(self):

        if self.split == "train":
            cache_file = f"{self.cache_dir}/sentiment.train.tokenized.pt"
            data_file_0 = f"{self.data_dir}/sentiment.train.0"
            data_file_1 = f"{self.data_dir}/sentiment.train.1"
        
        if self.split == 'dev':
            cache_file = f"{self.cache_dir}/sentiment.dev.tokenized.pt"
            data_file_0 = f"{self.data_dir}/sentiment.dev.0"
            data_file_1 = f"{self.data_dir}/sentiment.dev.1"

        if self.split == 'test':
            cache_file = f"{self.cache_dir}/sentiment.test.tokenized.pt"
            data_file_0 = f"{self.data_dir}/sentiment.test.0"
            data_file_1 = f"{self.data_dir}/sentiment.test.1"

        os.makedirs(self.cache_dir, exist_ok=True)
        if not os.path.exists(cache_file): 
            self._prepare_data(cache_file=cache_file, data_file_0=data_file_0, data_file_1=data_file_1, create_labels=True)

        if not self.prepare_data:
            logger.info(f"Loading {self.split} tokenization from cache [{cache_file}] ...")
            self.data = torch.load(cache_file)
        
    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, index):
        return {
            'input_ids': self.data['input_ids'][index], 
            'attention_mask': self.data['attention_mask'][index],
            'labels': self.data['labels'][index] 
        }

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'roberta-base', 
        use_fast=True, 
    )
    dataset = OriginalYelpDataset2(
        split='train', 
        tokenizer=tokenizer, 
        preprocess_kind='original', 
        max_seq_len=110, 
        # prepare_data=True
    )

    print(dataset.__getitem__(0))