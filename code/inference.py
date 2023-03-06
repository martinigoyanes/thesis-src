import os
import pytorch_lightning as pl
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler
from data_modules import ParaDetoxDM
from models import BARTdetox

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