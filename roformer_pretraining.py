# pretrain_roformer.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    RoFormerConfig,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, set_seed
)
from models.modeling_roformer import RoFormerForMaskedLM
import nltk
from nltk.tokenize import sent_tokenize

set_seed(42)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# --- 1) Load corpora (English Wikipedia + BookCorpusOpen variants on the Hub)
wiki  = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
books = load_dataset("lucadiliello/bookcorpusopen", split="train")
wiki = wiki.rename_columns({"text": "raw_text"}) if "text" in wiki.column_names else wiki
if "raw_text" not in wiki.column_names:
    wiki = wiki.map(lambda x: {"raw_text": x.get("text", "")})
if "text" in books.column_names and "raw_text" not in books.column_names:
    books = books.map(lambda x: {"raw_text": x["text"]})
corpus = concatenate_datasets([wiki, books])

# --- 2) Sentence splitting (keeps things roughly comparable to BERT-style packing)
def split_doc(ex):
    sents = [s.strip() for s in sent_tokenize(ex["raw_text"]) if s.strip()]
    return {"sentences": sents}
corpus = corpus.map(split_doc, remove_columns=corpus.column_names, num_proc=4)
corpus = corpus.filter(lambda x: len(x["sentences"])>0)

from transformers import AutoTokenizer

# define tokenizer BEFORE using it
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", use_fast=True)

def pack_examples_batched(batch, tokenizer, max_length):
    # batch["sentences"] is List[List[str]]
    sentences = []
    for doc_sents in batch["sentences"]:
        for s in doc_sents:
            if not isinstance(s, str):
                s = " ".join(s)
            sentences.append(s)

    texts, cur, cur_len = [], [], 0
    for s in sentences:
        cur.append(s)
        cur_len += len(s.split())
        if cur_len > max_length // 2:
            texts.append(" ".join(cur))
            cur, cur_len = [], 0
    if cur:
        texts.append(" ".join(cur))

    return tokenizer(texts, truncation=True, max_length=max_length)

def build_stream(max_len, tok):
    return corpus.map(
        pack_examples_batched,
        batched=True,
        remove_columns=["sentences"],
        num_proc=4,
        fn_kwargs={"tokenizer": tok, "max_length": max_len},  # <-- pass tokenizer here
    )

train_128 = build_stream(128, tokenizer)
train_512 = build_stream(512, tokenizer)

# --- 4) RoFormer config/model (Base-like)
config = RoFormerConfig(
    vocab_size=tokenizer.vocab_size,   # 30522 (WordPiece)
    hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072,
    max_position_embeddings=512, type_vocab_size=2, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
)
model = RoFormerForMaskedLM(config)

collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)

# --- 5) Phase 1: 900k steps @128 (accumulate to global batch≈256)
args_128 = TrainingArguments(
    output_dir="roformer_base_pretrain",
    max_steps=900_000,
    per_device_train_batch_size=32, gradient_accumulation_steps=8,  # ~256 global
    learning_rate=1e-4, weight_decay=0.01, warmup_steps=10_000, lr_scheduler_type="linear",
    bf16=torch.cuda.is_available(), logging_steps=100, save_steps=10_000, save_total_limit=3,
    dataloader_num_workers=4, report_to="none"
)

trainer = Trainer(model=model, args=args_128, train_dataset=train_128, data_collator=collator)
trainer.train()
trainer.save_model("roformer_base_phase1_ckpt")

# --- 6) Phase 2: 100k steps @512
model = RoFormerForMaskedLM.from_pretrained("roformer_base_phase1_ckpt")
args_512 = TrainingArguments(
    output_dir="roformer_base_pretrain_512",
    max_steps=100_000,
    per_device_train_batch_size=16, gradient_accumulation_steps=16,  # still ~256 global
    learning_rate=1e-4, weight_decay=0.01, lr_scheduler_type="linear",
    bf16=torch.cuda.is_available(), logging_steps=100, save_steps=10_000, save_total_limit=3,
    dataloader_num_workers=4, report_to="none"
)
trainer = Trainer(model=model, args=args_512, train_dataset=train_512, data_collator=collator)
trainer.train()
trainer.save_model("roformer_base_final")
tokenizer.save_pretrained("roformer_base_final")
