#!/usr/bin/env python3
"""
Fine-tuning Llama 3.2 3B sur dataset Dioula — version AWS/CUDA
Compatible : transformers 4.44, peft 0.12, trl 0.9, bitsandbytes 0.43
"""

import json
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer


# ── Chargement du JSONL ───────────────────────────────────────
def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ── Main ──────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Fine-tune Llama 3.2 3B sur Dioula (AWS/CUDA)")
    p.add_argument("--model",      default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                   help="Modèle HuggingFace ou chemin local")
    p.add_argument("--train",      default="train.jsonl")
    p.add_argument("--valid",      default="valid.jsonl")
    p.add_argument("--output",     default="./adapters_aws",
                   help="Dossier de sortie des checkpoints")
    p.add_argument("--iters",      type=int,   default=800,
                   help="Nombre de steps (800 = équivalent run Mac)")
    p.add_argument("--batch-size", type=int,   default=2)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--lora-rank",  type=int,   default=8)
    p.add_argument("--lora-alpha", type=float, default=20.0)
    p.add_argument("--max-seq",    type=int,   default=2048)
    p.add_argument("--hf-token",   default=None,
                   help="Token HuggingFace pour les modèles privés/gated")
    args = p.parse_args()

    # ── Token HuggingFace ─────────────────────────────────────
    hf_kwargs = {}
    if args.hf_token:
        hf_kwargs["token"] = args.hf_token

    # ── Info GPU ──────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponible — ce script nécessite un GPU NVIDIA.")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU  : {gpu_name}")
    print(f"VRAM : {vram_gb:.1f} GB")

    # ── QLoRA : quantization 4-bit ────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Tokenizer ─────────────────────────────────────────────
    print(f"\nChargement tokenizer : {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        **hf_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Modèle ────────────────────────────────────────────────
    print(f"Chargement modèle   : {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        **hf_kwargs,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # ── LoRA ──────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
    )

    # ── Datasets ──────────────────────────────────────────────
    print(f"\nChargement données : {args.train} / {args.valid}")
    train_ds = Dataset.from_list(load_jsonl(args.train))
    valid_ds = Dataset.from_list(load_jsonl(args.valid))
    print(f"Train : {len(train_ds)} exemples")
    print(f"Valid : {len(valid_ds)} exemples")

    # ── Arguments d'entraînement ──────────────────────────────
    use_bf16 = torch.cuda.get_device_capability()[0] >= 8   # Ampere+
    training_args = TrainingArguments(
        output_dir=args.output,
        max_steps=args.iters,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        warmup_steps=50,
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_8bit",
        report_to="none",
        dataloader_pin_memory=False,
    )

    # ── Trainer ───────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        peft_config=lora_config,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=args.max_seq,
        packing=False,
    )

    # ── Lancement ─────────────────────────────────────────────
    print(f"\nDémarrage fine-tuning ({args.iters} steps)...\n")
    trainer.train()

    # ── Sauvegarde finale ─────────────────────────────────────
    final_path = Path(args.output) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"\nAdaptateurs sauvegardés dans : {final_path}")


if __name__ == "__main__":
    main()
