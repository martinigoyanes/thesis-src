from transformers import BartForConditionalGeneration
import torch
# from transformers import OpenAIGPTLMHeadModel
from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
from transformers import RobertaForSequenceClassification
from pytorch_pretrained_bert import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIAdam, cached_path
import pytorch_lightning as pl
import logging
logger = logging.getLogger(__name__)

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
        # https://huggingface.co/blog/how-to-generate
        preds = self.generate(
            inputs=batch["input_ids"],
            num_return_sequences=1,
            do_sample=True,
            max_length=128,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            num_beams=1
        )
        return preds

class BlindGST(pl.LightningModule):
    def __init__(self, model_name_or_path: str, num_special_tokens: int, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model_name_or_path = model_name_or_path
        self.model = OpenAIGPTLMHeadModel.from_pretrained(self.model_name_or_path)
        self.model.resize_token_embeddings(num_special_tokens + self.model.config.vocab_size)
        # self.pad_token_id = pad_token_id

    @staticmethod
    def add_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BlindGST")
        parser.add_argument("--model_name_or_path", type=str, default='openai-gpt')
        parser.add_argument("--weight_decay", type=float, default=0., help="Regularization parameter during training")
        parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
        parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for Adam optimizer")
        parser.add_argument("--warmup_steps", type=int, default=0, help="Number of steps for linear warmup")
        parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Maximum norm of gradients")
        return parent_parser

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        output = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            labels=batch['labels'], 
            return_dict=True
        )
        self.log("train_loss", output.loss, on_step=True, on_epoch=True, prog_bar = True, logger=True)
        return {'loss':output.loss}
    
    def validation_step(self, batch, batch_idx):
        output = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            labels=batch['labels'], 
            return_dict=True
        )
        self.log("val_loss", output.loss, on_step=True, on_epoch=True, prog_bar = True, logger=True)
        return {'loss':output.loss}

    def predict_step(self, batch, batch_idx):
        # https://huggingface.co/blog/how-to-generate
        preds = self.generate(
            inputs=batch["input_ids"],
            num_return_sequences=1,
            do_sample=True,
            max_length=128,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            num_beams=1
        )
        # Generate with padding -> https://github.com/huggingface/transformers/pull/7552#issue-497255933
        # preds = self.generate(
        #     inputs=batch["input_ids"],
        #     attention_mask=batch["attention_mask"],
        #     pad_token_id=self.pad_token_id,
        #     num_return_sequences=1,
        #     do_sample=True,
        #     max_length=128,
        #     top_k=50,
        #     top_p=0.95,
        #     temperature=0.7,
        #     num_beams=1
        # )
        return preds
    
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

class OriginalBlindGST(pl.LightningModule):
    def __init__(self, model_name_or_path: str, num_special_tokens: int, batch_size: int, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model_name_or_path = model_name_or_path
        self.model = OpenAIGPTLMHeadModel.from_pretrained(self.model_name_or_path, num_special_tokens=num_special_tokens)
        self.batch_size = batch_size

    @staticmethod
    def add_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("OriginalBlindGST")
        parser.add_argument("--model_name_or_path", type=str, default='openai-gpt')
        parser.add_argument('--max_grad_norm', type=int, default=1)
        parser.add_argument('--learning_rate', type=float, default=6.25e-5)
        parser.add_argument('--warmup_proportion', type=float, default=0.002)
        parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--num_train_epochs', type=int, default=1)
        return parent_parser

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        loss = self(batch['input_ids'], lm_labels=batch['lm_labels'])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar = True, logger=True)
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        loss = self(batch['input_ids'], lm_labels=batch['lm_labels'])
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar = True, logger=True)
        return {'loss':loss}

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # TODO
        # train_data_len = 443259 # len(sentiment.train)
        # num_train_optimization_steps = train_data_len * self.hparams.num_train_epochs // self.batch_size
        optimizer = OpenAIAdam(optimizer_grouped_parameters,
                            lr=self.hparams.learning_rate,
                            warmup=self.hparams.warmup_proportion,
                            max_grad_norm=self.hparams.max_grad_norm,
                            weight_decay=self.hparams.weight_decay,
                            t_total=self.trainer.estimated_stepping_batches)
        return [optimizer]
    

class SentimentRoBERTa(pl.LightningModule):
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model_name_or_path = model_name_or_path
        id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        label2id = {"NEGATIVE": 0, "POSITIVE": 1}
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_name_or_path, 
            num_labels=2,
            id2label=id2label,
            label2id=label2id
        )

    @staticmethod
    def add_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BlindGST")
        parser.add_argument("--model_name_or_path", type=str, default='roberta-base')
        parser.add_argument("--weight_decay", type=float, default=0., help="Regularization parameter during training")
        parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
        parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for Adam optimizer")
        parser.add_argument("--warmup_steps", type=int, default=100, help="Number of steps for linear warmup")
        parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Maximum norm of gradients")
        return parent_parser

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        output = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            labels=batch['labels'], 
            return_dict=True
        )
        self.log("train_loss", output.loss, on_step=True, on_epoch=True, prog_bar = True, logger=True)
        return {'loss':output.loss}
    
    def validation_step(self, batch, batch_idx):
        output = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            labels=batch['labels'], 
            return_dict=True
        )
        probs = torch.sigmoid(output.logits)
        preds = torch.argmax(probs, dim=1)
        self.log("val_loss", output.loss, on_step=True, on_epoch=True, prog_bar = True, logger=True)
        return {'loss': output.loss, 'preds': preds, 'labels': batch['labels']}

    def validation_epoch_end(self, outputs):
        preds, labels = [], []
        for out in outputs:
            preds += out['preds'].tolist()
            labels += out['labels'].tolist()

        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, fscore, _ = precision_recall_fscore_support(y_true=labels, y_pred=preds, average='binary')
        self.log("val_precision", precision, on_epoch=True, prog_bar = True, logger=True)
        self.log("val_recall", recall, on_epoch=True, prog_bar = True, logger=True)
        self.log("val_fscore", fscore, on_epoch=True, prog_bar = True, logger=True)

    def test_step(self, batch, batch_idx):
        output = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            labels=batch['labels'], 
            return_dict=True
        )
        probs = torch.sigmoid(output.logits)
        preds = torch.argmax(probs, dim=1)
        self.log("test_loss", output.loss, on_step=True, on_epoch=True, prog_bar = True, logger=True)
        return {'loss': output.loss, 'preds': preds, 'labels': batch['labels']}
    
    def test_epoch_end(self, outputs):
        preds, labels = [], []
        for out in outputs:
            preds += out['preds'].tolist()
            labels += out['labels'].tolist()
        
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, fscore, _ = precision_recall_fscore_support(y_true=labels, y_pred=preds, average='binary')
        self.log("test_precision", precision, on_epoch=True, prog_bar = True, logger=True)
        self.log("test_recall", recall, on_epoch=True, prog_bar = True, logger=True)
        self.log("test_fscore", fscore, on_epoch=True, prog_bar = True, logger=True)
        self._save_results(precision, recall, fscore)
        self._save_preds(preds, labels)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]
        
    def _save_results(self, precision, recall, fscore):
        out_dir = self.trainer.default_root_dir
        logger.info(f"Writint results to {out_dir}") 
        with open(f"{out_dir}/results.md", 'a') as res_file:
            res_file.writelines('PRECISION | RECALL | FSCORE \n')
            res_file.writelines('--------- | ------ | ------ \n')
            res_file.writelines(f'{precision:.4f}|{recall:.4f}|{fscore:.4f}\n')

    def _save_preds(self, preds, labels):
        out_dir = self.trainer.default_root_dir
        logger.info(f"Writint preds to {out_dir}") 
        with open(f"{out_dir}/preds.txt", 'a') as f:
            preds_txt = "Preds: " + " ".join(str(p) for p in preds) + "\n"
            labels_txt = "Labels: " + " ".join(str(l) for l in labels) + "\n"
            f.write(preds_txt)
            f.write(labels_txt)

if __name__ == "__main__":
    from data_modules import OriginalJigsawDM

    dm = OriginalJigsawDM(
        tokenizer_name_or_path='openai-gpt',
        max_seq_len=300, # Real max_len = 266 ----> 300
        batch_size=32,
        preprocess_kind='bert_best_head_removal'
    )
    dm.setup(stage="fit")

    model = OriginalBlindGST(
        model_name_or_path='openai-gpt',
        num_special_tokens=len(dm.special_tokens),
        batch_size=dm.batch_size
    )

    batch = dm.datasets['train'][:5]

    loss = model(batch['input_ids'], lm_labels=batch['lm_labels'])

    # For classifiers
    # probs = torch.sigmoid(output.logits)
    # preds = torch.argmax(probs, dim=1)

    # from sklearn.metrics import precision_recall_fscore_support
    # precision, recall, fscore, support = precision_recall_fscore_support(y_true=batch['labels'], y_pred=preds, average='binary')

    # print(precision)