import torch
import pytorch_lightning as pl
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler
from data_modules import ParaDetoxDM
from models import BARTdetox

if __name__ == "__main__":
    from argparse import  ArgumentParser

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument("--model_name_or_path", type=str, default='facebook/bart-base')
    parser.add_argument("--tokenizer_name_or_path", type=str, default='facebook/bart-base')
    parser.add_argument("--batch_size", type=int, default=32) # I made this up
    parser.add_argument("--max_seq_len", type=int, default=None) # Dont use for inference
    parser.add_argument("--out_dir", type=str, default='.')
    args = parser.parse_args()

    pl.seed_everything(44)

    model = BARTdetox(args)
    dm = ParaDetoxDM(args)

    # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        fast_dev_run=True,
        deterministic=True,
        # profiler="advanced",
    )
    trainer.fit(model, datamodule=dm)