import os
from pytorch_lightning.callbacks import ModelCheckpoint
from fsspec.core import url_to_fs
import pytorch_lightning as pl
import logging 

logger = logging.getLogger(__name__)

class HfModelCheckpoint(ModelCheckpoint):
	def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
		super()._save_checkpoint(trainer, filepath)
		model = trainer.lightning_module.model
		tokenizer = trainer.datamodule.tokenizer
		if getattr(model, "save_pretrained", None) and getattr(tokenizer, "save_pretrained", None):
			dir_name = "/".join(filepath.split('/')[:-1])
			logger.info(f"Saving model and tokenizer in {dir_name}...")
			model_path = f"{dir_name}/model"
			tokenizer_path = f"{dir_name}/tokenizer"
			if trainer.is_global_zero:
				model.save_pretrained(model_path)
				tokenizer.save_pretrained(tokenizer_path)
	
	# https://github.com/Lightning-AI/lightning/pull/16067
	def _remove_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
		super()._remove_checkpoint(trainer, filepath)
		hf_save_dir = filepath+".dir"
		if trainer.is_global_zero:
			fs, _ = url_to_fs(hf_save_dir)
			if fs.exists(hf_save_dir):
				fs.rm(hf_save_dir, recursive=True)

def split_refs(test_file, out_dir):
	'''
	Takes in test file containing source and target and splits into 2 files, 
	one with input for the model and other with expected output
	'''
	with open(test_file, 'r') as f:
		lines = [line.strip() for line in f.readlines()]

	# get source style
	src_style = test_file.split("/")[-1].split(".")[-1]
	# src_style = int(test_file[-1])
	ref_in_f, ref_out_f = f'{out_dir}/reference.{src_style}.in', f'{out_dir}/reference.{src_style}.out'

	if os.path.exists(ref_in_f) or os.path.exists(ref_out_f):
		print(f"Files already exist:\n\t-{ref_out_f} or\n\t-{ref_in_f}")

	for line in lines:
		in_text, out_text = [], []

		tokens = line.split()
		start_token_idx = tokens.index('<START>')

		in_text = tokens[:start_token_idx+1] # Include <START>	
		out_text = tokens[start_token_idx+1:-1] # We dont want the <START> or <END> token

		with open(ref_in_f, 'a') as f:
			in_text = " ".join(in_text) + "\n"
			f.write(in_text)

		with open(ref_out_f, 'a') as f:
			out_text = " ".join(out_text) + "\n"
			f.write(out_text)

if __name__ == "__main__":
	out_dir = '/home/martin/Documents/Education/Master/thesis/project/thesis-src/data/jigsaw/bert_best_head_removal'

	test_file = '/home/martin/Documents/Education/Master/thesis/project/thesis-src/data/jigsaw/bert_best_head_removal/test.neutral'

	split_refs(test_file=test_file, out_dir=out_dir)