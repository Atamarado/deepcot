import os
import pickle
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd

from finetune_glue_roformer import get_tokenized_cache_path, get_roformer_config, tokenize_fn, runtime_loop, get_model, ROFORMER_FOLDER

def runtime_experiment(config, batch_size=1):
	args = get_roformer_config()

	# Set config
	arg_vars = vars(args)
	for k, v in config.items():
		arg_vars[k] = v

	match args.model:
		case 'roformer':
			args.model_name = ROFORMER_FOLDER + "/roformer_base_final"
		case 'modernbert':
			args.model_name = 'answerdotai/ModernBERT-base'
		case 'fnet':
			args.model_name = 'google/fnet-base'
		case _:
			raise Exception("Incorrect model name")

	args.batch_size = batch_size

	tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

	raw = load_dataset("glue", args.task)
	cols_to_remove = raw["train"].column_names

	# Tokenize
	def _tokenize_runtime(batch):
		return tokenize_fn(args.task, tok, batch, use_CLS=False, window_size=-1, min_length=-1, max_length=-1, attention_mask=False)
	tokenized = raw.map(
		_tokenize_runtime,
		batched=True,
		remove_columns=cols_to_remove,
		load_from_cache_file=True,
		cache_file_names=get_tokenized_cache_path(args, 'runtime'),
		desc="Building sliding windows",
	)

	# Merge all samples (from validation and test sets) into one
	sep_token = tok.added_tokens_encoder['[SEP]']

	subsets = list(tokenized.data.keys())
	subsets.remove('train')

	token_sequence = []
	first_sequence = True

	for subset in subsets:
		for i in range(len(tokenized[subset])):
			input_ids = tokenized[subset][i]['input_ids']

			if first_sequence:
				first_sequence = False
			else:
				token_sequence.append(sep_token)
			token_sequence += input_ids

	reduction_tokens = 16
	assert batch_size <= reduction_tokens
	ratio_reduction = reduction_tokens // batch_size

	if ratio_reduction > 1:
		token_sequence = token_sequence[:len(token_sequence)//ratio_reduction]

	module_batch = len(token_sequence) % batch_size
	if module_batch > 0:
		token_sequence = token_sequence[:-module_batch]
	token_sequence = [{'input_ids': torch.tensor(token_sequence).view(batch_size, -1)}]
	# data_loader = torch.utils.data.DataLoader(token_sequence, batch_size=1, collate_fn=collator)

	model = get_model(args, args.task, load_dir = False)
	total_time, _ = runtime_loop(args, model, token_sequence, max_batches=-1, tqdm_window=True)

	iterations = len(token_sequence[0]['input_ids'][0]) - args.window_size
	return total_time, iterations

def get_runtime_path(out_dir, name, wsize, batch_size=1):
	path = f"{name}_{wsize}"
	if batch_size > 1:
		path += f"_{batch_size}"
	path += ".pkl"
	path = os.path.join(out_dir, path)
	return path

if __name__ == '__main__':
	out_dir = 'runtimes'
	os.makedirs(out_dir, exist_ok=True)

	experiments = {
		'base_roformer': {'deepcot': False},
		'deepcot_roformer': {'deepcot': True, 'deepcot_train': True, 'forward_steps_train': True},
		'base_soft': {'deepcot': False, 'reduced_attention': True},
		'deepcot_soft': {'deepcot': True, 'deepcot_train': True, 'forward_steps_train': True, 'reduced_attention': True},
		'modernbert': {'deepcot': False, 'model': 'modernbert'},
		'fnt': {'deepcot': False, 'model': 'fnet'},
	}

	res = []

	batch_sizes = [1, 16]

	for batch_size in batch_sizes:
		window_sizes = [2**i for i in range(4, 10)] + [i*1000 for i in range(1, 11)]
		try:
			for wsize in window_sizes:
				for name, exp in experiments.items():
					exp['window_size'] = wsize
					exp['max_length'] = max(512, wsize+1)

					path = get_runtime_path(out_dir, name, wsize, batch_size)

					res_row = {"model": name, "window_size": wsize, }

					if os.path.exists(path):
						print(f"{wsize}_{name} already available. Skipping...")
						with open(path, "rb") as f:
							out_obj = pickle.load(f)
					else:
						print(f"{wsize}_{name}: Starting experiment")
						running_time, iterations = runtime_experiment(exp, batch_size)

						out_obj = {
							"running_time": running_time,
							"iterations": iterations,
						}

						with open(path, "wb") as f:
							pickle.dump(out_obj, f)

					res_row.update(out_obj)
					res.append(res_row)
		except:
			pass

	df = pd.DataFrame(res)
	df.to_csv('runtimes.csv', index=False)
