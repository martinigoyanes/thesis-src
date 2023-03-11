from transformers import BartForConditionalGeneration
import torch
# from transformers import OpenAIGPTLMHeadModel, get_linear_schedule_with_warmup
from pytorch_pretrained_bert import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, OpenAIAdam, cached_path
import pytorch_lightning as pl

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
    def __init__(self, model_name_or_path: str, num_special_tokens: int, pad_token_id: int,  out_dir: str, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model_name_or_path = model_name_or_path
        self.model = OpenAIGPTLMHeadModel.from_pretrained(self.model_name_or_path)
        self.model.resize_token_embeddings(num_special_tokens + self.model.config.vocab_size)
        self.pad_token_id = pad_token_id
        self.out_dir = out_dir

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
            attention_mask=batch["attention_mask"],
            pad_token_id=self.pad_token_id,
            num_return_sequences=1,
            do_sample=True,
            max_length=128,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            num_beams=1
        )
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
        train_data_len = 443259 # len(sentiment.train)
        num_train_optimization_steps = train_data_len * self.hparams.num_train_epochs // self.batch_size
        optimizer = OpenAIAdam(optimizer_grouped_parameters,
                            lr=self.hparams.learning_rate,
                            warmup=self.hparams.warmup_proportion,
                            max_grad_norm=self.hparams.max_grad_norm,
                            weight_decay=self.hparams.weight_decay,
                            t_total=num_train_optimization_steps)
        return [optimizer]
    