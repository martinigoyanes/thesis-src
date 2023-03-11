import os
import torch
import logging
import pytorch_lightning as pl
from data_modules import ParaDetoxDM, YelpDM
from models import BARTdetox, BlindGST

logger = logging.getLogger(__name__)


def save_preds(preds, out_dir, tokenizer):
    path = f"{out_dir}/preds.txt"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    for pred_ids in preds:
        prediction_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        with open(path, 'a') as f: f.write('\n'.join(prediction_texts)+'\n')

def main(args):

    # Setup Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    dict_args = vars(args)

    # pick datamodule
    if args.datamodule_name == "YelpDM":
        dm = YelpDM(**dict_args)
    # pick model
    if args.model_name == "BlindGST":
        model = BlindGST.load_from_checkpoint(args.checkpoint_path)

    # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pl.seed_everything(44)
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=1,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        # fast_dev_run=True,
        # deterministic=True,
        # limit_train_batches=10,
        # limit_val_batches=10,
        # profiler="advanced",
    )

    preds = trainer.predict(model=model, datamodule=dm)
    save_preds(preds=preds, out_dir=args.default_root_dir, tokenizer=dm.tokenizer)

if __name__ == "__main__":
    from argparse import  ArgumentParser

    # Resolve handler is key to be able to overrid trainer args with model/datamodule specific args
    parser = ArgumentParser(conflict_handler='resolve')
    parser = pl.Trainer.add_argparse_args(parser)

    # figure out which model to use
    ckpt_root_dir = '/Midgard/home/martinig/thesis-src/jobs/gst/training/364455'
    ckpt_path = f'{ckpt_root_dir}/lightning_logs/version_364455/checkpoints/epoch=0-step=13852.ckpt'
    parser.add_argument("--model_name", type=str, default="BlindGST", help="BlindGST")
    parser.add_argument("--datamodule_name", type=str, default="YelpDM", help="YelpDM")
    parser.add_argument("--preprocess_kind", type=str, default="bert_best_head_removal", help="Kind of preprocessing of data:\n - bert_best_head_removal")
    parser.add_argument("--checkpoint_path", type=str, default=ckpt_path, help="Path to checkpoint from trainer.fit()")
    parser.add_argument("--default_root_dir", type=str, default=ckpt_root_dir, help="Directory to store run logs and ckpts")


    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # let the model add what it wants
    if temp_args.model_name == "BlindGST":
        parser = BlindGST.add_specific_args(parser)
    # let the datamodule add what it wants
    if temp_args.datamodule_name == "YelpDM":
        parser = YelpDM.add_specific_args(parser)

    args = parser.parse_args()

    # train
    main(args)