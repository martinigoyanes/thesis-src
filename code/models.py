from transformers import BartForConditionalGeneration
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
        preds = self.generate(
            inputs=batch["input_ids"],
            num_return_sequences=1,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=3.0,
            num_beams=10
        )
        return preds