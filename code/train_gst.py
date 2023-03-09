import torch
import pytorch_lightning as pl
import logging
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler
from data_modules import YelpDM
from models import BlindGST

logger = logging.getLogger(__name__)


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
        model = BlindGST(num_special_tokens=len(dm.special_tokens), **dict_args)

    # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"

    trainer = pl.Trainer(
        args,
        max_epochs=1,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        fast_dev_run=True,
        deterministic=True,
        # profiler="advanced",
    )
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    # TODO: Check if tokenizer vocab and model embedding sizes match
    from argparse import  ArgumentParser

    # Resolve handler is key to be able to overrid trainer args with model/datamodule specific args
    parser = ArgumentParser(conflict_handler='resolve')
    parser = pl.Trainer.add_argparse_args(parser)

    # figure out which model to use
    parser.add_argument("--model_name", type=str, default="BlindGST", help="BlindGST")
    parser.add_argument("--datamodule_name", type=str, default="YelpDM", help="YelpDM")
    parser.add_argument("--preprocess_kind", type=str, default="bert_best_head_removal", help="Kind of preprocessing of data:\n - bert_best_head_removal")


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