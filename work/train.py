# -*- coding: utf-8 -*-
"""
Training:
- CLM training for Model-A (tiny) and Model-B (small)
- Windows + NVIDIA GPU: fp16 + gradient checkpointing enabled
"""
import gc
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, LlamaConfig, LlamaForCausalLM,
    DataCollatorForLanguageModeling, TrainingArguments, Trainer, set_seed
)
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
DATA = WORK / "data"
MODEL_ROOT = WORK / "model"

set_seed(42)

def train_one(variant: str):
    assert variant in ("A", "B")
    train_txt = DATA / ("train_A_2MB.txt" if variant == "A" else "train_B_10MB.txt")
    eval_txt  = DATA / ("eval_A.txt" if variant == "A" else "eval_B.txt")
    vocab_dir = MODEL_ROOT / variant / "tokenizer"
    model_dir = MODEL_ROOT / variant

    tok = AutoTokenizer.from_pretrained(str(vocab_dir))
    tok.pad_token = tok.eos_token

    raw = load_dataset("text", data_files={"train":[str(train_txt)], "validation":[str(eval_txt)]})
    max_length = 64 if variant == "A" else 128
    def tokenize(ex):
        return tok(ex["text"], truncation=True, padding=True,
                   max_length=max_length, return_overflowing_tokens=True, return_length=True)
    ds = raw.map(tokenize, batched=True, remove_columns=raw["train"].column_names)

    if variant == "A":
        cfg = dict(hidden_size=384, n_layers=6, n_heads=6, interm=4*384, context=512,
                   lr=3e-4, epochs=5, per_dev_bs=16, grad_acc=4)
    else:
        cfg = dict(hidden_size=512, n_layers=8, n_heads=8, interm=4*512, context=512,
                   lr=2e-4, epochs=5, per_dev_bs=16, grad_acc=8)

    config = LlamaConfig(
        vocab_size=len(tok),
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["n_layers"],
        intermediate_size=cfg["interm"],
        num_attention_heads=cfg["n_heads"],
        bos_token_id=tok.convert_tokens_to_ids("<|endoftext|>"),
        eos_token_id=tok.convert_tokens_to_ids("<|endoftext|>"),
        pad_token_id=tok.convert_tokens_to_ids("<pad>"),
        max_position_embeddings=cfg["context"]
    )
    model = LlamaForCausalLM(config)
    model.gradient_checkpointing_enable()

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    args = TrainingArguments(
        output_dir=str(model_dir),
        overwrite_output_dir=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["per_dev_bs"],
        gradient_accumulation_steps=cfg["grad_acc"],
        learning_rate=cfg["lr"],
        warmup_steps=200,
        lr_scheduler_type="cosine",
        logging_steps=25,
        report_to="none",
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model, tokenizer=tok, args=args,
        data_collator=collator,
        train_dataset=ds["train"], eval_dataset=ds["validation"]
    )
    trainer.train()

    final_dir = model_dir / "final"
    final_dir.mkdir(exist_ok=True)
    trainer.save_model(str(final_dir))
    pd.DataFrame(trainer.state.log_history).to_csv(model_dir / "losses.csv", index=False)

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final_dir, vocab_dir

if __name__ == "__main__":
    print(">>> Training Model A")
    A_final, A_tok = train_one("A")
    print(f"A saved to {A_final}")

    print(">>> Training Model B")
    B_final, B_tok = train_one("B")
    print(f"B saved to {B_final}")
