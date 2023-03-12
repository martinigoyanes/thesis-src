import torch
import pytorch_lightning as pl
import logging
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler
from data_modules import YelpDM
from data_modules import YelpDM2
from data_modules import OriginalYelpDM
from models import BlindGST
from models import OriginalBlindGST
from models import SentimentRoBERTa
from utils import HfModelCheckpoint

logger = logging.getLogger(__name__)

# TODO: Save fine-tuned tokenizer and model -> https://github.com/Lightning-AI/lightning/issues/3096
    # Uncomment line to save model when I stop using pytorch_bert
    # load_from_checkpoint() does not load tokenizer, neither it gets saved
# TODO: Load fine-tuned tokenizer and model -> 
# TODO: Lm_labels start after <START> (dont include it)
# TODO: Use all data

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
    if args.datamodule_name == "YelpDM2":
        dm = YelpDM2(**dict_args)
    if args.datamodule_name == "OriginalYelpDM":
        dm = OriginalYelpDM(**dict_args)
    # pick model
    if args.model_name == "BlindGST":
        model = BlindGST(num_special_tokens=len(dm.special_tokens), **dict_args)
    if args.model_name == "OriginalBlindGST":
        model = OriginalBlindGST(num_special_tokens=len(dm.special_tokens), **dict_args)
    if args.model_name == "SentimentRoBERTa":
        model = SentimentRoBERTa(**dict_args)


    # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pl.seed_everything(44)

    checkpoint_callback = HfModelCheckpoint()
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback],
        max_epochs=1,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        # fast_dev_run=True,
        # deterministic=True,
        # limit_train_batches=10,
        # limit_val_batches=10,
        # profiler="advanced",
    )
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    from argparse import  ArgumentParser

    # Resolve handler is key to be able to overrid trainer args with model/datamodule specific args
    parser = ArgumentParser(conflict_handler='resolve')
    parser = pl.Trainer.add_argparse_args(parser)

    # figure out which model to use
    # parser.add_argument("--model_name", type=str, default="BlindGST", help="BlindGST")
    # parser.add_argument("--model_name", type=str, default="OriginalBlindGST", help="OriginalBlindGST")
    parser.add_argument("--model_name", type=str, default="SentimentRoBERTa", help="SentimentRoBERTa")
    # parser.add_argument("--datamodule_name", type=str, default="YelpDM", help="YelpDM")
    # parser.add_argument("--datamodule_name", type=str, default="OriginalYelpDM", help="OriginalYelpDM")
    parser.add_argument("--datamodule_name", type=str, default="YelpDM2", help="YelpDM2")
    # parser.add_argument("--preprocess_kind", type=str, default="bert_best_head_removal", help="Kind of preprocessing of data:\n - bert_best_head_removal\n - original")
    parser.add_argument("--preprocess_kind", type=str, default="original", help="Kind of preprocessing of data:\n - bert_best_head_removal\n - original")
    parser.add_argument("--default_root_dir", type=str, default=".", help="Directory to store run logs and ckpts")


    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # let the model add what it wants
    if temp_args.model_name == "BlindGST":
        parser = BlindGST.add_specific_args(parser)
    if temp_args.model_name == "OriginalBlindGST":
        parser = OriginalBlindGST.add_specific_args(parser)
    if temp_args.model_name == "SentimentRoBERTa":
        parser = SentimentRoBERTa.add_specific_args(parser)
    # let the datamodule add what it wants
    if temp_args.datamodule_name == "OriginalYelpDM":
        parser = OriginalYelpDM.add_specific_args(parser)
    if temp_args.datamodule_name == "YelpDM":
        parser = YelpDM.add_specific_args(parser)
    if temp_args.datamodule_name == "YelpDM2":
        parser = YelpDM2.add_specific_args(parser)

    args = parser.parse_args()

    # train
    main(args)