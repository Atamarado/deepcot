#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
finetune_glue_roformer.py

Fine-tune RoFormer on a GLUE task and evaluate.
- If a trained checkpoint exists in --out, loads it and evaluates (unless --force_train).
- Otherwise, trains then saves a final checkpoint, and evaluates.

Usage examples:
  # Fine-tune from your pre-trained RoFormer and save to runs/roformer_mnli
  python finetune_glue_roformer.py --task mnli --model_name roformer_base_final --out runs/roformer_mnli

  # Later: just evaluate the saved checkpoint
  python finetune_glue_roformer.py --task mnli --out runs/roformer_mnli

  # Or evaluate a specific checkpoint dir
  python finetune_glue_roformer.py --task mnli --checkpoint runs/roformer_mnli/checkpoint-5000
"""

import argparse
import json

from transformers.trainer_utils import PredictionOutput

import time
from pathlib import Path
from typing import Optional, Tuple
import pickle
import sys
import os

import numpy as np
from ptflops import get_model_complexity_info

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
    __version__ as HF_VER,
    TrainerCallback, AutoModelForSequenceClassification,
)

import continual_dev as co
from models import RoFormerForSequenceClassification

ROFORMER_FOLDER = 'roformer_models'

# ----------------------------- Utils -----------------------------------------

def find_trained_model_dir(args, out_dir: str) -> Optional[str]:
    """
    Return a directory that contains a trained model:
      - the final `out_dir` if it has pytorch_model.bin or model.safetensors
      - otherwise, the latest 'checkpoint-*' subdir if present
    """
    p = Path(out_dir)
    if (p / "pytorch_model.bin").exists() or (p / "model.safetensors").exists():
        return str(p)
    ckpts = [d for d in p.glob("checkpoint-*")
             if (d / "pytorch_model.bin").exists() or (d / "model.safetensors").exists()]

    path_train_logger = get_metrics_path(Path(args.out), args, extension='.pkl', name='train_logger')
    if not os.path.exists(path_train_logger) and args.deepcot:
        return None

    if not os.path.exists(get_metrics_path(Path(args.out), args)):
        return None

    if ckpts:
        ckpts.sort(key=lambda d: int(d.name.split("-")[-1]))
        return str(ckpts[-1])
    return None


def select_pred_samples(pred_output: PredictionOutput, positions: list[int]) -> PredictionOutput:
    """
    Returns a new PredictionOutput containing only the entries at the given positions.
    """
    positions = np.array(positions)

    # Slice predictions and labels using the given positions
    new_predictions = pred_output.predictions[positions]
    new_label_ids = pred_output.label_ids[positions] if pred_output.label_ids is not None else None

    # Metrics are typically aggregate values (not per sample),
    # so you can either keep them as is or return an empty dict
    new_metrics = {}

    return PredictionOutput(
        predictions=new_predictions,
        label_ids=new_label_ids,
        metrics=new_metrics
    )

def evaluate_loop(args, model, trainer, tok, subset, path_pred, window_size, compute_metrics):
    with co.call_mode("forward_steps"):
        if args.model == 'roformer':
            model.clean_state(always_clean=True)
        predictions_output = trainer.predict(tok[subset])

    seq_lens = [len(tok[subset][i]["input_ids"]) for i in range(len(tok[subset]))]
    to_save = {
        "prediction_output": predictions_output,
        "seq_len": seq_lens,
    }

    # Save the predictions for future analysis
    with open(path_pred, "wb") as f:
        pickle.dump(to_save, f)

    # Evaluate with all the samples
    eval_full = compute_metrics(predictions_output)
    eval_full = {f"{subset}_full_{k}": v for k, v in eval_full.items()}
    print(json.dumps({"eval_split": subset, **eval_full}, indent=2))

    # If necessary, evaluate with the samples that are >= args.window_size
    selected_idxs = [i for i, x in enumerate(seq_lens) if x >= window_size]
    if window_size > 0 and len(selected_idxs) > 0 and len(seq_lens) > len(selected_idxs):
        predictions_output = select_pred_samples(predictions_output, selected_idxs)
        eval_win = compute_metrics(predictions_output)
        eval_win = {f"{subset}_win_{k}": v for k, v in eval_win.items()}
        print(json.dumps({**eval_win}, indent=2))
        eval_full.update(eval_win)
    return eval_full

class EmptyCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        return control

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def is_regression(task: str) -> bool:
    return task.lower() == "stsb"

def glue_eval_splits(task: str) -> Tuple[str, Optional[str]]:
    """Return (eval_primary, eval_secondary) split names; MNLI has two."""
    if task.lower() == "mnli":
        return "validation_matched", "validation_mismatched"
    return "validation", None

def glue_test_splits(task: str) -> Tuple[str, Optional[str]]:
    """Return (test_primary, test_secondary) split names; MNLI has two."""
    if task.lower() == "mnli":
        return "test_matched", "test_mismatched"
    return "test", None

def get_out_folder(args):
    out = ROFORMER_FOLDER + f"/{args.model}_glue_{args.task}"
    if args.window_size > 0:
        out += f"_{args.window_size}"
    if args.use_CLS:
        out += "_CLS"
    if args.deepcot_train:
        out += "_deepcottrain"
        if args.forward_steps_train:
            out += "_forwardstepstrain"
    if args.reduced_attention:
        out += "_reducedatt"
    out += f"_{args.seed}"
    return out

def get_metrics_path(task_dir, args, extension='.json', name='metrics'):
    if args.deepcot:
        name += "_deepcot"

    name += extension
    return task_dir / name

PATH_TOKENIZED_CACHE = 'tokenized_caches/'
os.makedirs(PATH_TOKENIZED_CACHE, exist_ok=True)
def get_tokenized_cache_path(args, tokenize_type):
    task = args.task
    window_size = args.window_size

    assert tokenize_type in ['base', 'minwindow', 'eval', 'all_eval', 'runtime'], 'Incorrect tokenize type'
    assert tokenize_type != 'minwindow' or window_size > 0, 'runtime requires establishing a window size > 0'

    path = PATH_TOKENIZED_CACHE + f'{args.model}_glue_{task}_{tokenize_type}'
    if tokenize_type != 'runtime' and window_size > 0:
        path += f'_w{window_size}'
    elif tokenize_type not in ['minwindow', 'runtime']:
        path += f'_l{args.max_length}'

    if tokenize_type == 'base' and args.use_CLS:
        path += '_CLS'

    if tokenize_type == 'all_eval' and args.deepcot:
        path += '_deepcot'

    path_dict = {}
    dataset_list = ['train'] + list(glue_eval_splits(task)) + list(glue_test_splits(task))
    for dataset_name in dataset_list:
        path_dict[dataset_name] = path + f"_{dataset_name}.pth"

    return path_dict

def get_num_labels(task):
    match task:
        case 'stsb':
            return 1
        case 'mnli':
            return 3
        case _:
            return 2

def get_model(args, task, load_dir):
    config = AutoConfig.from_pretrained(args.model_name, num_labels=get_num_labels(task))
    if is_regression(task):
        config.problem_type = "regression"
    config.window_size = args.window_size
    config.deepcot = args.deepcot
    config.use_CLS = args.use_CLS

    config.force_window_size = True
    config.reduced_attention = args.reduced_attention

    if args.model == 'roformer':
        model_fn = RoFormerForSequenceClassification
    else:
        model_fn = AutoModelForSequenceClassification

    # Instantiate model
    if load_dir:
        config.merge_qk = False
        print(f"[INFO] Loading fine-tuned checkpoint: {load_dir}")
        model = model_fn.from_pretrained(load_dir, config=config)
    else:
        config.merge_qk = args.reduced_attention
        print(f"[INFO] No fine-tuned checkpoint found (or --force_train set). "
              f"Initializing from base: {args.model_name}")
        model = model_fn.from_pretrained(args.model_name, config=config)

    if args.model == 'fnet' and args.max_length > 512:
        model.fnet.embeddings.position_ids = torch.arange(args.max_length).expand((1, -1))
        model.fnet.embeddings.position_embeddings = torch.nn.Embedding(args.max_length, 768)
        model.fnet.embeddings.token_type_ids = torch.zeros(args.max_length, dtype=torch.long).expand((1, -1))

    model = model.to('cuda')
    return model

def _last_window(seq, ws):
    n = len(seq)
    if n < ws:
        return [seq]
    return [seq[-ws:]]

def _windows(seq, ws, st):
    """Return sliding windows of size ws with step st over seq."""
    n = len(seq)
    if n == 0:
        return []
    if ws <= 0:
        return [seq]
    if n <= ws:
        return [seq]
    out, i = [], 0
    last_start = max(0, n - ws)
    while i <= last_start:
        out.append(seq[i:i + ws])
        i += st
    return out

def tokenize_fn(task, tok, batch, window_size, use_CLS, max_length, window_step=1, min_length=0, last_window_only=False, attention_mask=True):
    """
    Tokenizes the text inputs in different ways
    Args:
        task: The specific GLUE task
        tokenizer: The Autotokenizer to use
        batch: The batch to tokenize
        window_size: The window size to generate. If window_size <= 0, a single window will be generated, with the
            length of the sequence.
        max_length: Maximum allowed length of an input. This MUST only be used when window_size <= 0. If max_length <= 0,
            any length will be considered valid.
        window_step: The window step that is used (i.e., how many tokens to skip after generating a window). Default=1
        min_length: Minimum allowed length of an input. Default=0.
        last_window_only: Boolean input. It can only be set to True if window_size > 0. Instead of returning all
            window combinations, it just returns the last possible attention window.
    Returns:
        The tokenized batch.
    """
    assert not last_window_only or window_size > 0
    assert not(max_length > 0 and window_size > 0)

    sent1_key, sent2_key = TASK_TO_KEYS[task]

    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []  # only for pair tasks
    labels_list = []

    texts1 = batch[sent1_key]
    is_pair = sent2_key is not None
    texts2 = batch[sent2_key] if is_pair else None

    # IMPORTANT: Never pass through original 'label' as-is.
    labels_src = batch.get("label", None)

    if max_length > 0:
        kwargs_enc = {"padding": False, "truncation": True, "max_length": max_length}
    else:
        kwargs_enc = {"padding": False, "truncation": False}

    enc1 = tok(texts1, **kwargs_enc)
    if is_pair:
        enc2 = tok(texts2, **kwargs_enc)

    for i, ids1 in enumerate(enc1["input_ids"]):
        if not use_CLS:
            ids1 = ids1[1:]

        if is_pair:
            ids2 = enc2["input_ids"][i][1:]

            tti = [0] * len(ids1) + [1] * len(ids2)
            ids1 = ids1 + ids2

        if len(ids1) < min_length:
            continue

        if len(ids1) > max_length:
            ids1 = ids1[:max_length]
            if is_pair:
                tti = tti[:max_length]

        if window_size > 0:
            if last_window_only:
                chunks = _last_window(ids1, window_size)
                if is_pair:
                    tti_chunks = _last_window(tti, window_size)
            else:
                chunks = _windows(ids1, window_size, window_step)
                if is_pair:
                    tti_chunks = _windows(tti, window_size, window_step)
        else:
            # Build a single chunk
            chunks = [ids1]
            if is_pair:
                tti_chunks = [tti]

        # append per chunk
        for j, ch in enumerate(chunks):
            input_ids_list.append(ch)
            if attention_mask:
                attention_mask_list.append([1] * len(ch))
            if is_pair:
                token_type_ids_list.append(tti_chunks[j])
            if labels_src is not None:
                labels_list.append(labels_src[i])

    # Build output dict
    out = {
        "input_ids": input_ids_list,
    }
    if attention_mask:
        out["attention_mask"] = attention_mask_list
    if is_pair:
        out["token_type_ids"] = token_type_ids_list
    if labels_src is not None:
        out["labels"] = labels_list

    return out

def runtime_loop(args, model, data_loader, max_batches=64, tqdm_window=False):
    total_time = 0.
    num_windows = 0

    if len(data_loader) == 0:
        return 0., 0

    with torch.no_grad():
        enough_inputs = False
        while not enough_inputs:
            for i, batch in tqdm(enumerate(data_loader)):
                if i > max_batches > 0:
                    break

                cuda_batch = {}
                for k, v in batch.items():
                    if k == 'attention_mask':
                        if args.model == 'fnet':
                            continue
                        v = torch.ones_like(v)  # We don't have masked elements in continual inference
                    cuda_batch[k] = v.to(model.device)
                batch = cuda_batch

                # Warmup phase
                if args.model == 'roformer':
                    model.clean_state(always_clean=False)

                window_size = args.window_size if args.window_size > 0 else batch['input_ids'].shape[1]
                if args.deepcot:
                    warmup_batch = {}
                    for k in batch.keys():
                        if k == 'labels':
                            continue
                        warmup_batch[k] = batch[k][:, :window_size - 1]

                    with co.call_mode("forward_steps"):
                        model(**warmup_batch)

                # Prepare the subsequent windows that we'll feed to the model with running time measuring
                with co.call_mode("forward_step"):
                    if tqdm_window:
                        range_for = tqdm(range(window_size - 1, batch['input_ids'].shape[1]))
                    else:
                        range_for = range(window_size - 1, batch['input_ids'].shape[1])

                    for j in range_for:
                        window = {}
                        for k in batch.keys():
                            if k == 'labels':
                                continue
                            if args.deepcot:
                                window[k] = batch[k][:, j:j + 1]
                            else:
                                window[k] = batch[k][:, j - window_size + 1:j + 1]

                        if window is not None:
                            start = time.time()
                            outputs = model(**window)
                            end = time.time()
                            total_time += end - start
                            num_windows += 1

            if num_windows >= max_batches:
                enough_inputs = True
    return total_time, num_windows

# ----------------------------- Main ------------------------------------------
def get_roformer_config():
    ap = argparse.ArgumentParser("Roformer config", add_help=False)

    ap.add_argument("--task", type=str, required=False,
                    choices=list(TASK_TO_KEYS.keys()),
                    help="GLUE task name")
    ap.add_argument("--model_name", type=str, default=None,
                    help="Base or fine-tuned model name/path")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Specific fine-tuned checkpoint directory to load for evaluation")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force_train", action="store_true",
                    help="Ignore existing checkpoint(s) and train anyway")
    ap.add_argument("--force_eval", action="store_true",
                    help="Ignore existing results and evaluate anyway")
    ap.add_argument("--force_running_time", action="store_true",
                    help="Ignore existing results and compute the running time anyway")
    ap.add_argument("--deepcot", action="store_true", help="Perform DeepCoT evaluation.")
    ap.add_argument("--window_size", type=int, default=-1, help="Window size for DeepCoT evaluation. Use window_size<=0 for the full attention window")
    ap.add_argument("--window_step", type=int, default=1, help="Step size for DeepCoT evaluation")
    ap.add_argument("--use_CLS", action="store_true", help="Use CLS for fine-tuning and evaluation. If not CLS is used, inference happens only in the last token")
    ap.add_argument("--deepcot_train", action="store_true", help="Train using DeepCoT models (or load the corresponding checkpoint)")
    ap.add_argument("--reduced_attention", action="store_true", help="Uses the reduced attention that has the guarantees described in the paper description."
                                                                     "The changes include using SOFT instead of softmax, removing LayerNorm operations and the non-linear ff activations")
    ap.add_argument("--forward_steps_train", action="store_true", help="Train in forward_steps mode (useful for the DeepCoT train method)")
    ap.add_argument("--model", type=str, default='roformer', help="Model to use. Options: [roformer, modernbert, fnet] (mind that most of the configurations are only available for 'roformer')")

    args = ap.parse_args()

    if args.model_name is None:
        match args.model:
            case 'roformer':
                args.model_name = ROFORMER_FOLDER+"/roformer_base_final"
            case 'modernbert':
                args.model_name = 'answerdotai/ModernBERT-base'
            case 'fnet':
                args.model_name = 'google/fnet-base'
            case _:
                raise Exception("Incorrect model name")

    return args


def main():
    args = get_roformer_config()

    assert args.window_step == 1, "window_step must be 1 for now"
    assert not (args.use_CLS and args.deepcot_train), "You cannot train using CLS and DeepCoT"
    assert not args.deepcot_train or args.deepcot, "You must use DeepCoT when performing deepcot_train"
    assert not(args.forward_steps_train and not args.deepcot_train), "You shouldn't train in forward_steps non-DeepCoT models"

    set_seed(args.seed)

    args.out = get_out_folder(args)
    os.makedirs(args.out, exist_ok=True)

    # ---------------- Data ----------------
    task = args.task.lower()
    raw = load_dataset("glue", task)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # --- window config ---
    W = args.window_size  # >0 → manual sliding; <=0 → passthrough
    S = args.window_step
    if W > 0:
        assert 1 <= S <= W, "window_step must be in [1, window_size]"

    def _tokenize_fn(batch):
        return tokenize_fn(task, tok, batch, use_CLS=args.use_CLS, window_size=W, max_length=-1 if W > 0 else args.max_length)

    def _tokenize_fn_eval(batch):
        return tokenize_fn(task, tok, batch, use_CLS=False, window_size=W, min_length=W, max_length=-1 if W>0 else args.max_length, last_window_only=W>0)

    def _tokenize_fn_eval_all(batch):
        return tokenize_fn(task, tok, batch, use_CLS=False, window_size=W, min_length=-1, max_length=-1 if W>0 else args.max_length, last_window_only=W>0)

    def _tokenize_fn_eval_all_deepcot(batch):
        return tokenize_fn(task, tok, batch, use_CLS=False, window_size=-1, min_length=-1, max_length=-1)

    def _tokenize_full_window_fwd_steps(batch):
        return tokenize_fn(task, tok, batch, use_CLS=False, window_size=-1, min_length=W, max_length=-1, attention_mask=False)

    # Remove ALL original columns; only keep our new ones
    cols_to_remove = raw["train"].column_names
    collator = DataCollatorWithPadding(tokenizer=tok)

    # ---------------- Model ----------------
    # Choose what to load:
    load_dir: Optional[str] = None
    if args.checkpoint:
        load_dir = args.checkpoint
    else:
        # If not forcing training, prefer any existing trained model under --out
        if not args.force_train:
            load_dir = find_trained_model_dir(args, args.out)

    model = get_model(args, task, load_dir)
    # ---------------- Metrics ----------------
    import evaluate
    metric = evaluate.load("glue", task)

    match task:
        case 'cola':
            metric = [metric, evaluate.load("f1"), evaluate.load("accuracy")]
        case 'stsb':
            from mae_score import MAEScore
            metric = [metric, MAEScore()]
        case 'sst2' | 'qnli' | 'rte' | 'wnli':
            metric = [metric, evaluate.load("f1")]

    def compute_metrics(p):
        preds = p.predictions
        if is_regression(task):
            preds = np.squeeze(preds)
        else:
            preds = np.argmax(preds, axis=1)

        if type(metric) == list:
            res = {}
            for met in metric:
                res_met = met.compute(predictions=preds, references=p.label_ids)
                res.update(res_met)
            return res

        return metric.compute(predictions=preds, references=p.label_ids)

    # ---------------- TrainingArguments ----------------
    # HF v4.56 uses eval_strategy (not evaluation_strategy).
    eval_split, eval_split_mm = glue_eval_splits(task)
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 and args.model != 'fnet'

    targs = TrainingArguments(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="pearson" if task == "stsb" else (
            "matthews_correlation" if task == "cola" else "accuracy"
        ),
        greater_is_better=True if task in ("stsb",) else True,
        logging_steps=100,
        report_to="none",
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available() and args.model != 'fnet',
    )

    if args.reduced_attention:
        targs.warmup_ratio = 0.1
        targs.max_grad_norm = 0.5
        targs.leaning_rate = 1e-6

    metrics_path = get_metrics_path(Path(args.out), args)

    def get_tokenized_train(args):
        if args.deepcot_train and args.forward_steps_train:
            tokenized_train = raw.map(
                _tokenize_fn_eval_all_deepcot,
                batched=True,
                remove_columns=cols_to_remove,
                load_from_cache_file=True,
                cache_file_names=get_tokenized_cache_path(args, 'all_eval'),
                desc="Building sliding windows (eval)",
            )
        else:
            tokenized_train = raw.map(
                _tokenize_fn,
                batched=True,
                remove_columns=cols_to_remove,
                load_from_cache_file=True,
                cache_file_names=get_tokenized_cache_path(args, 'base'),
                desc="Building sliding windows (train)",
            )
        return tokenized_train

    def get_tokenized_eval_all(args):
        if args.deepcot:
            tokenized_eval = raw.map(
                _tokenize_fn_eval_all_deepcot,
                batched=True,
                remove_columns=cols_to_remove,
                load_from_cache_file=True,
                cache_file_names=get_tokenized_cache_path(args, 'all_eval'),
                desc="Building sliding windows (eval)",
            )
        else:
            tokenized_eval = raw.map(
                _tokenize_fn_eval_all,
                batched=True,
                remove_columns=cols_to_remove,
                load_from_cache_file=True,
                cache_file_names=get_tokenized_cache_path(args, 'all_eval'),
                desc="Building sliding windows (eval)",
            )
        return tokenized_eval

    # ---------------- Train (if needed) ----------------
    tokenized_eval = None
    if args.force_train or not load_dir:
        if args.deepcot and not args.deepcot_train:
            raise Exception("You MUST set args.deepcot_train if you want to train in DeepCoT mode")
        if args.deepcot and args.use_CLS:
            raise Exception("DeepCoT cannot be trained with CLS tokens.")

        tokenized_train = get_tokenized_train(args)
        tokenized_eval = get_tokenized_eval_all(args)

        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=tokenized_train["train"],
            eval_dataset=tokenized_eval[eval_split],
            processing_class=tok,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=[EmptyCacheCallback()],
        )

        class FlopsWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                x = x.long()
                return self.model(x)

        print("[INFO] Starting training…")
        if args.deepcot_train and args.forward_steps_train:
            with co.call_mode("forward_steps"):
                if args.model == 'roformer':
                    model.clean_state(always_clean=True)

                # macs, params = get_model_complexity_info(FlopsWrapper(model), (208,), as_strings=False, print_per_layer_stat=False)
                # flops = macs // 2
                # print("Model FLOPs {}. Model params: {}".format(flops, params))
                trainer.train()
        else:
            # macs, params = get_model_complexity_info(FlopsWrapper(model), (208,), as_strings=False, print_per_layer_stat=False)
            # flops = macs // 2
            # print("Model FLOPs {}. Model params: {}".format(flops, params))
            trainer.train()
        path_train_logger = get_metrics_path(Path(args.out), args, extension='.pkl', name='train_logger')
        # Save the train_logger
        with open(path_train_logger, "wb") as f:
            pickle.dump(trainer.state.log_history, f)

        print("[INFO] Saving final checkpoint to:", args.out)
        trainer.save_model(args.out)
        tok.save_pretrained(args.out)

    # ---------------- Evaluate ----------------
    if not args.force_eval and not args.force_train and load_dir and metrics_path.exists():
        metrics = None
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        eval_primary = {}
        eval_mm = {}
        for k, v in metrics.items():
            if k.startswith(eval_split):
                eval_primary[k] = v
            elif eval_split_mm is not None and k.startswith(eval_split_mm):
                eval_mm[k] = v

    else:
        if tokenized_eval is None:
            tokenized_eval = get_tokenized_eval_all(args)

        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=None,
            eval_dataset=tokenized_eval[eval_split],
            processing_class=tok,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=[EmptyCacheCallback()],
        )

        print(f"[INFO] Evaluating on split: {eval_split}")
        path_pred = get_metrics_path(Path(args.out), args, extension='.pred', name='predictions')
        eval_primary = evaluate_loop(args, model, trainer, tokenized_eval, eval_split, path_pred, args.window_size, compute_metrics)

        # MNLI has two validation splits; evaluate the other as well.
        if task == "mnli" and eval_split_mm is not None:
            path_pred = get_metrics_path(Path(args.out), args, extension='.pred', name='predictions_mm')
            eval_mm = evaluate_loop(args, model, trainer, tokenized_eval, eval_split_mm, path_pred, args.window_size, compute_metrics)

    # ---------------- Running time ----------------
    total_time = 0.
    n_windows_runtime = 0.
    if args.force_running_time:
        print(f"[INFO] Computing running time")

        # Set the last transformer layer to only output the last token every time, and allow for warmup to happen in DeepCoT models
        if args.model == 'roformer':
            model.set_config(last_token_output = True, force_window_size=False)

        tokenized_running_time = raw.map(
            _tokenize_full_window_fwd_steps,
            batched=True,
            remove_columns=cols_to_remove,
            load_from_cache_file=True,
            cache_file_names=get_tokenized_cache_path(args, 'minwindow'),
            desc="Building sliding windows",
        )

        eval_dataloader = torch.utils.data.DataLoader(tokenized_running_time[eval_split], batch_size=args.batch_size, collate_fn=collator)
        total_time, n_windows_runtime = runtime_loop(args, model, eval_dataloader, max_batches=512)
        print(f"Inference time: {total_time}")
    else:
        try:
            metrics = None
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            total_time = metrics['inference_time']
        except:
            total_time = 0.

    # ---------------- FLOPS ----------------
    if args.window_size > 0 and args.model == 'roformer':
        flops = model.flops()
    else:
        flops = 0

    # ---------------- Write metrics ----------------
    all_metrics = {"hf_version": HF_VER, "inference_time": total_time, "flops": flops, 'n_windows_runtime': n_windows_runtime}
    all_metrics.update(eval_primary)
    if task == "mnli" and eval_split_mm is not None:
        all_metrics.update(eval_mm)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[INFO] Metrics written to {metrics_path}")

if __name__ == "__main__":
    main()
