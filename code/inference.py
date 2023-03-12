import os
import torch
import logging
import pytorch_lightning as pl
from data_modules import ParaDetoxDM
from data_modules import YelpDM
from data_modules import OriginalYelpDM
from models import BlindGST
from models import BARTdetox
from models import OriginalBlindGST

logger = logging.getLogger(__name__)


def save_preds(preds, out_dir, tokenizer):
    path = f"{out_dir}/preds.txt"
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    pred_texts = []
    # Decode All the beam indexes to till <END> token only and convert into sentence
    for pred_ids in preds:
        try:
            end_index = pred_ids.index(tokenizer.special_tokens["<END>"])
        except ValueError:
            end_index = len(pred_ids)
        pred_texts.append(tokenizer.decode(pred_ids[:end_index]))

    with open(path, 'w') as f: f.write('\n'.join(pred_texts)+'\n')


def predict(texts, beam_width=3, vocab_length=40483, tokenizer=None, device=None, model=None):
    """
    This function decodes sentences using Beam Seach. 
    It will output #sentences = beam_width. This function works on a single example.
    
    ref_text : string : Input sentence
    beam_width : int : Width of the output beam
    vocab_length : int : Size of the Vocab after adding the special tokens
    """
    
    
    sm = torch.nn.Softmax(dim=-1) # To calculate Softmax over the final layer Logits

    pred_ids = []
    for ref_text in texts:
        done = [False for i in range(beam_width)] # To track which beams are already decoded
        stop_decode = False
        tokens = tokenizer.tokenize(ref_text) # Tokenize the input text
        
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokens) # Convert tokens to ids
        index_tokens = [indexed_tokens for i in range(beam_width)] # Replication of Input ids for all the beams

        #index_tokens = [indexed_tokens for i in range(beam_width)]
        torch_tensor = torch.tensor(index_tokens).to(device)
        beam_indexes = [[] for i in range(beam_width)] # indexes of the current decoded beams
        best_scoes = [0 for i in range(beam_width)] # A list of lists to store Probability values of each decoded token of best beams
        count = 0
        logger.info(f"Predicted {len(pred_ids)} sentences")
        while count < model.config.n_positions and not stop_decode:
            if count == 0: # For the first step when only one sentence is availabe
                with torch.no_grad():
                    # Calculate output probability distribution over the Vocab,
                    preds = sm(model(torch_tensor)) #  shape = [beam_bidth, len(input_sen)+1,Vocab_length]
                top_v, top_i = preds[:,-1,:].topk(beam_width) # Fatch top indexes and it's values
                [beam_indexes[i].append(top_i[0][i].tolist()) for i in range(beam_width)] # Update the Beam indexes
                # Update the best_scores, for first time just add the topk values directly
                for i in range(beam_width):
                    best_scoes[i] = top_v[0][i].item()
                count += 1
            else: # After first step
                # Prepare the current_state by concating original input and decoded beam indexes
                current_state = torch.cat((torch_tensor, torch.tensor(beam_indexes).to(device)), dim=1)
                # Prediction on the current state
                with torch.no_grad():
                    preds = sm(model(current_state))
                # Multiply new probability predictions with corresponding best scores
                # Total socres = beam_width * Vocab_Size
                flatten_score = (preds[:,-1,:]*torch.tensor(best_scoes).to(device).unsqueeze(1)).view(-1)
                # Fatch the top scores and indexes 
                vals, inx = flatten_score.topk(beam_width)
                # best_score_inx saves the index of best beams after multiplying the probability of new prediction
                best_scoes_inx = (inx / vocab_length).tolist()
                best_scoes = vals.tolist()
                # Unflatten the index 
                correct_inx = (inx % vocab_length).tolist()
                
                # Check if done for all the Beams
                for i in range(beam_width):
                    if correct_inx[i] == tokenizer.special_tokens["<END>"]:
                        done[i] = True
                # Update the best score for each the current Beams
                for i in range(beam_width):
                    if not done[i]:
                        best_scoes[i] = vals.tolist()[i]
                # Check is All the Beams are Done
                if (sum(done) == beam_width):
                    stop_decode = True
                # Prepapre the new beams
                temp_lt=[0 for i in range(beam_width)]
                for i,x in enumerate(best_scoes_inx):
                    temp_lt[i] = beam_indexes[i] + [correct_inx[i]]
                # Update the Beam indexes
                beam_indexes = temp_lt
                del temp_lt
                count += 1
                if current_state.size(1) >= 128:
                    break
    
        pred_ids += beam_indexes
    return pred_ids

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
    if args.datamodule_name == "OriginalYelpDM":
        dm = OriginalYelpDM.load_from_checkpoint(args.checkpoint_path)
    # pick model
    if args.model_name == "BlindGST":
        model = BlindGST.load_from_checkpoint(args.checkpoint_path)
    if args.model_name == "OriginalBlindGST":
        model = OriginalBlindGST.load_from_checkpoint(args.checkpoint_path)

    # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pl.seed_everything(44)
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=1,
        accelerator="auto",
        devices=model.device
        # fast_dev_run=True,
        # deterministic=True,
        # limit_train_batches=10,
        # limit_val_batches=10,
        # profiler="advanced",
    )

    # preds = trainer.predict(model=model, datamodule=dm)
    logger.info(f"Will output inference predictions to {dm.hparams.default_root_dir}")
    dm.setup(stage='predict')
    preds = predict(dm.datasets['test'], beam_width=1, vocab_length=40483, tokenizer=dm.tokenizer, device=model.device, model=model.model)
    save_preds(preds=preds, out_dir=dm.hparams.default_root_dir, tokenizer=dm.tokenizer)

if __name__ == "__main__":
    from argparse import  ArgumentParser

    # Resolve handler is key to be able to overrid trainer args with model/datamodule specific args
    parser = ArgumentParser(conflict_handler='resolve')
    parser = pl.Trainer.add_argparse_args(parser)

    # figure out which model to use
    # parser.add_argument("--model_name", type=str, default="BlindGST", help="BlindGST")
    parser.add_argument("--model_name", type=str, default="OriginalBlindGST", help="OriginalBlindGST")
    # parser.add_argument("--datamodule_name", type=str, default="YelpDM", help="YelpDM")
    parser.add_argument("--datamodule_name", type=str, default="OriginalYelpDM", help="OriginalYelpDM")
    parser.add_argument("--preprocess_kind", type=str, default="bert_best_head_removal", help="Kind of preprocessing of data:\n - bert_best_head_removal")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint from trainer.fit()")


    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # let the model add what it wants
    if temp_args.model_name == "BlindGST":
        parser = BlindGST.add_specific_args(parser)
    if temp_args.model_name == "OriginalBlindGST":
        parser = OriginalBlindGST.add_specific_args(parser)
    # let the datamodule add what it wants
    if temp_args.datamodule_name == "YelpDM":
        parser = YelpDM.add_specific_args(parser)
    if temp_args.datamodule_name == "OriginalYelpDM":
        parser = OriginalYelpDM.add_specific_args(parser)

    args = parser.parse_args()

    # train
    main(args)