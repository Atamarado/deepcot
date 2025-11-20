#!/usr/bin/env python
# run_glue_roformer.py
# Fine-tunes RoFormer on all GLUE tasks via finetune_glue_roformer.py
# and aggregates results into a pandas DataFrame.

import argparse
import json
import os
import pickle
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import torch
from datasets import load_dataset
from torch import nn, Tensor
from transformers import AutoTokenizer, DataCollatorWithPadding

from finetune_glue_roformer import get_out_folder, get_metrics_path, runtime_loop, tokenize_fn, \
	get_tokenized_cache_path, glue_eval_splits

# We're not using RTE and WNLI
GLUE_TASKS = ["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli"]#, "rte", "wnli"]

MAIN_METRIC = {
	"cola": "f1",
	"sst2": "f1",
	"mrpc": "f1",
	"qqp": "f1",
	"stsb": "mae_score",
	"mnli": "accuracy",
	"qnli": "f1",
	"rte":  "f1",
	"wnli": "f1",
}

PAPER_METRIC = {
	"cola": "matthews_correlation",
	"sst2": "accuracy",
	"mrpc": "f1",
	"qqp": "f1",
	"stsb": "spearmanr",
	"mnli": "accuracy",
	"qnli": "accuracy",
	"rte": "accuracy",
	"wnli": "accuracy",
}

# 1x
WINDOW_SIZES = {
	"cola": 12,
	"sst2": 24,
	"mrpc": 52,
	"qqp": 30,
	"stsb": 30,
	"mnli": 38,
	"qnli": 50,
}

def get_window_sizes(factor):
	if factor <= 0:
		return {key: -1 for key, _ in WINDOW_SIZES.items()}
	return {key: int(value * factor) for key, value in WINDOW_SIZES.items()}

# 2x get_window_sizes(2)
# 4x get_window_sizes(4)
# original get_window_sizes(-1)

def run_one_task(
	args,
) -> Optional[int]:
	"""Run finetuning/eval for a task and return (returncode)."""

	cmd = [
		args.python or "python",
		args.script,
		"--task", args.task,
		"--epochs", str(args.epochs),
		"--batch_size", str(args.batch_size),
		"--lr", str(args.lr),
		"--seed", str(args.seed),
		"--window_size", str(args.window_size),
		"--model", str(args.model),
	]
	if args.model_name:
		cmd.append("--model_name")
		cmd.append(args.model_name)
	if args.force_train:
		cmd.append("--force_train")
	if args.force_eval:
		cmd.append("--force_eval")
	if args.force_running_time:
		cmd.append("--force_running_time")
	if args.reduced_attention:
		cmd.append("--reduced_attention")
	if args.deepcot:
		cmd.append("--deepcot")
		cmd.append("--deepcot_train")
		cmd.append('--forward_steps_train')
	if args.deepcot_train:
		cmd.append("--deepcot_train")
	if args.forward_steps_train:
		cmd.append('--forward_steps_train')

	print(f"\n[RUN] {args.task.upper()} -> {cmd}")
	try:
		res = subprocess.run(cmd, check=False)
		return res.returncode
	except Exception as e:
		print(f"[ERROR] Failed to run {args.task}: {e}")
		return None

def read_metrics(task_dir: Path, args) -> Dict:
	"""Load metrics.json written by finetune_glue_roformer.py."""
	metrics_path = get_metrics_path(task_dir, args)
	if not metrics_path.exists():
		print(f"[WARN] {str(metrics_path)} not found in {task_dir}")
		return {}
	with open(metrics_path, "r", encoding="utf-8") as f:
		return json.load(f)

def maybe_read_trainer_state(task_dir: Path) -> Dict:
	"""Try to read best checkpoint info from trainer_state.json (if present)."""
	st_path = task_dir / "trainer_state.json"
	if not st_path.exists():
		return {}
	try:
		with open(st_path, "r", encoding="utf-8") as f:
			data = json.load(f)
		keep = {}
		if "best_model_checkpoint" in data:
			keep["best_model_checkpoint"] = data["best_model_checkpoint"]
		if "best_metric" in data:
			keep["best_metric"] = data["best_metric"]
		return keep
	except Exception:
		return {}

def add_metric(row, task, metric, name):
	if task == "mnli":
		# Prefer matched + mismatched main metric and compute a simple average if both exist
		m1 = row.get(f"validation_matched_full_{metric}")
		m2 = row.get(f"validation_mismatched_full_{metric}")
		if m1 is not None and m2 is not None:
			row[f"{name}_full"] = (m1 + m2) / 2.0
		else:
			row[f"{name}_full"] = m1 if m1 is not None else m2

		try:
			m1 = row.get(f"validation_matched_win_{metric}")
			m2 = row.get(f"validation_mismatched_win_{metric}")
			if m1 is not None and m2 is not None:
				row[f"{name}_win"] = (m1 + m2) / 2.0
			else:
				row[f"{name}_win"] = m1 if m1 is not None else m2
		except:
			pass
	else:
		row[f"{name}_full"] = row.get(f"validation_full_{metric}")
		try:
			row[f"{name}_win"] = row.get(f"validation_win_{metric}")
		except:
			pass

	return row

def extract_task_row(task: str, metrics: Dict) -> Dict:
	"""
	Build a flat row with the most relevant metrics.
	Our finetune script writes keys like:
	  "validation_eval_accuracy", "validation_eval_loss", ...
	  For MNLI also "validation_mismatched_eval_accuracy", ...
	"""
	row = {"task": task}

	# Primary/secondary split names as used by finetune_glue_roformer.py
	if task == "mnli":
		prefixes = ["validation_matched", "validation_mismatched"]
	else:
		prefixes = ["validation"]

	# Pull common fields for each prefix if present
	for pref in prefixes:
		for k, v in metrics.items():
			if k.startswith(pref + "_"):
				row[k] = v

	row = add_metric(row, task, MAIN_METRIC[task], "main_metric")
	# row = add_metric(row, task, PAPER_METRIC[task], "paper_metric")

	# Keep eval loss if present
	for pref in prefixes:
		if f"{pref}_eval_loss" in row:
			row["eval_loss" + ("" if task != "mnli" else f"_{pref.split('_')[-1]}")] = row[f"{pref}_eval_loss"]

	# Also preserve STS-B's secondary metric for visibility
	if task == "stsb":
		if "validation_eval_spearmanr" in row:
			row["spearmanr_full"] = row["validation_eval_full_spearmanr"]
			try:
				row["spearmanr_win"] = row["validation_eval_win_spearmanr"]
			except:
				pass

	if "inference_time" in metrics.keys():
		row["inference_time"] = metrics["inference_time"]
	if "flops" in metrics.keys():
		row["flops"] = metrics["flops"]

	if "n_windows_runtime" in metrics.keys():
		row["n_windows_runtime"] = metrics["n_windows_runtime"]

	return row

def add_row_args(row, args, model):
	row["use_CLS"] = args.use_CLS
	row["window_size"] = args.window_size
	row["deepcot"] = args.deepcot
	row["task"] = args.task
	row["model"] = model

	return row

class CustomIdentity(nn.Identity):
	def __init__(self, *args: Any, **kwargs: Any):
		super().__init__(*args, **kwargs)
		self.device = 'cuda'

	def clean_state(self, always_clean=False):
		pass

	def forward(self, *args, **kwargs) -> Tensor:
		return torch.empty(1)

def get_num_windows_runtime(args):
	args.out = get_out_folder(args)
	num_windows_path = get_metrics_path(Path(args.out), args, extension='.pth', name='windows_runtime')

	if os.path.exists(num_windows_path):
		return pickle.load(open(num_windows_path, 'rb'))

	if args.model_name is None:
		match args.model:
			case 'roformer':
				args.model_name = "roformer_models/roformer_base_final"
			case 'modernbert':
				args.model_name = 'answerdotai/ModernBERT-base'
			case 'fnet':
				args.model_name = 'google/fnet-base'
			case _:
				raise Exception("Incorrect model name")

	model = CustomIdentity() # Create dummy model

	tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

	raw = load_dataset("glue", args.task)
	cols_to_remove = raw["train"].column_names
	collator = DataCollatorWithPadding(tokenizer=tok)

	def _tokenize_full_window_fwd_steps(batch):
		return tokenize_fn(args.task, tok, batch, use_CLS=False, window_size=-1, min_length=args.window_size, max_length=-1,
		                   attention_mask=False)

	tokenized_running_time = raw.map(
		_tokenize_full_window_fwd_steps,
		batched=True,
		remove_columns=cols_to_remove,
		load_from_cache_file=True,
		cache_file_names=get_tokenized_cache_path(args, 'minwindow'),
		desc="Building sliding windows",
	)

	eval_split, eval_split_mm = glue_eval_splits(args.task)
	eval_dataloader = torch.utils.data.DataLoader(tokenized_running_time[eval_split], batch_size=args.batch_size,
	                                              collate_fn=collator)

	_, num_windows = runtime_loop(args, model, eval_dataloader,tqdm_window=False)

	with open(num_windows_path, 'wb') as f:
		pickle.dump(num_windows, f)
	return num_windows

def main():
	p = argparse.ArgumentParser()
	p.add_argument("--script", type=str, default="finetune_glue_roformer.py",
				   help="Path to finetune_glue_roformer.py")
	p.add_argument("--model_name", type=str, default=None,
				   help="Base/fine-tuned model path for initialization")
	p.add_argument("--epochs", type=int, default=3)
	p.add_argument("--batch_size", type=int, default=32)
	p.add_argument("--lr", type=float, default=3e-5)
	p.add_argument("--seed", type=int, default=0)
	p.add_argument("--force_train", action="store_true",
	                help="Ignore existing checkpoint(s) and train anyway")
	p.add_argument("--force_eval", action="store_true",
	                help="Ignore existing results and evaluate anyway")
	p.add_argument("--force_running_time", action="store_true",
	                help="Ignore existing results and compute the running time anyway")
	p.add_argument("--tasks", type=str, default="all",
				   help="Comma-separated subset (e.g. 'sst2,mrpc') or 'all'")
	p.add_argument("--python", type=str, default=None,
				   help="Python interpreter to use (default: current PATH)")
	p.add_argument("--save_csv", type=str, default="glue_results.csv",
				   help="Filename for aggregated CSV (saved under out_root)")
	p.add_argument("--use_CLS", action="store_true",
	                help="Use CLS for fine-tuning and evaluation. If not CLS is used, inference happens only in the last token")
	p.add_argument("--deepcot_train", action="store_true",
	                help="Train using DeepCoT models (or load the corresponding checkpoint)")
	p.add_argument("--reduced_attention", action="store_true",
	                help="Uses the reduced attention that has the guarantees described in the paper description."
	                     "The changes include using SOFT instead of softmax, removing LayerNorm operations and the non-linear ff activations")
	p.add_argument("--forward_steps_train", action="store_true",
	                help="Train in forward_steps mode (useful for the DeepCoT train method)")
	p.add_argument("--model", type=str, default='roformer',
	                help="Model to use. Options: [roformer, modernbert, fnet] (mind that most of the configurations are only available for 'roformer')")

	args = p.parse_args()

	tasks: List[str] = GLUE_TASKS if args.tasks.lower() == "all" else [t.strip().lower() for t in args.tasks.split(",")]

	rows = []
	for model in ['roformer', 'soft', 'modernbert', 'fnet']:
		if model == 'soft':
			args.model = 'roformer'
			args.reduced_attention = True
		else:
			args.model = model
			args.reduced_attention = False
		for ws_factor in [0.5, 1, 2]:
			window_sizes = get_window_sizes(ws_factor)
			for task in tasks:
				args.task = task
				args.window_size = window_sizes[task]

				deepcot_options = [True, False] if model in ['roformer', 'soft'] else [False]
				for deepcot in deepcot_options:
					args.deepcot = deepcot
					args.deepcot_train = deepcot
					args.forward_steps_train = deepcot

					if args.force_train or args.force_eval or args.force_running_time:
						run_one_task(args)

					task_dir = Path(get_out_folder(args))

					# Collect metrics (even if return code was non-zero we still try)
					metrics = read_metrics(task_dir, args)
					if len(metrics) == 0:
						continue
					row = extract_task_row(task, metrics)
					# Merge optional trainer_state info
					row.update(maybe_read_trainer_state(task_dir))

					row = add_row_args(row, args, model)

					# row['n_windows_runtime'] = get_num_windows_runtime(args)
					if row['n_windows_runtime'] == 0:
						row['tps'] = 0
					else:
						row['tps'] = row['n_windows_runtime'] / row['inference_time']

					rows.append(row)

	df = pd.DataFrame(rows).set_index("task")

	print("\n=== Aggregated GLUE Results ===")
	print(df)

	csv_path = str(args.save_csv)
	df.to_csv(csv_path)
	print(f"\n[OK] Saved CSV to: {csv_path}")

if __name__ == "__main__":
	main()
