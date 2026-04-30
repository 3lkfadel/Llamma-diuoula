"""
finetune_aws.py
===============
Fine-tuning Llama 3.2 sur GPU AWS avec Unsloth (QLoRA 4-bit).
Optimisé pour 20 000+ itérations sur dataset Dioula.

Setup AWS (une seule fois) :
    pip install unsloth
    pip install torch --index-url https://download.pytorch.org/whl/cu121

Usage :
    python finetune_aws.py
    python finetune_aws.py --iters 20000 --model unsloth/Llama-3.2-3B-Instruct
"""

import json
import argparse
import os
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────
MODEL_ID    = "unsloth/Llama-3.2-3B-Instruct"   # téléchargement auto depuis HuggingFace
DATA_DIR    = Path("data")
OUTPUT_DIR  = Path("llama-dioula-aws")
MAX_SEQ_LEN = 512    # suffisant pour nos phrases Dioula courtes

LORA_CONFIG = {
    "r":             32,      # rang LoRA — 32 est bon pour une langue
    "lora_alpha":    64,      # = 2 × r, standard
    "lora_dropout":  0.05,
    "target_modules": [       # couches à adapter
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}

TRAIN_CONFIG = {
    "per_device_train_batch_size": 8,    # GPU AWS peut gérer 8 (vs 2 sur M1)
    "gradient_accumulation_steps": 4,   # batch effectif = 32
    "warmup_steps":                200,
    "max_steps":                   20000,
    "learning_rate":               2e-4,
    "lr_scheduler_type":           "cosine",  # descend progressivement
    "weight_decay":                0.01,
    "fp16":                        False,
    "bf16":                        True,      # A10G/V100 supporte bf16
    "logging_steps":               100,
    "eval_steps":                  500,
    "save_steps":                  1000,
    "output_dir":                  str(OUTPUT_DIR / "checkpoints"),
    "evaluation_strategy":         "steps",
    "save_total_limit":            5,         # garde les 5 meilleurs checkpoints
    "load_best_model_at_end":      True,
    "report_to":                   "none",    # mettre "wandb" si tu veux tracker
}

SYSTEM_PROMPT = """Tu es un assistant expert en langue Dioula (Jula), parlée principalement en Côte d'Ivoire, au Burkina Faso et au Mali.

Tes capacités :
- Traduire du français vers le Dioula et du Dioula vers le français
- Expliquer la grammaire Dioula (structure SOV, auxiliaires aspectuels, tons)
- Répondre en Dioula si l'utilisateur te parle en Dioula

Points clés de la grammaire Dioula :
- Structure de base : Sujet + Auxiliaire + Objet + Verbe (SOV)
- Auxiliaire 'bɛ' = présent affirmatif | 'tɛ' = présent négatif
- Auxiliaire 'ye' = passé affirmatif | 'ma' = passé négatif
- Auxiliaire 'bɛna' = futur affirmatif | 'tɛna' = futur négatif
- Langue tonale : les tons changent le sens des mots

Réponds toujours de manière précise et naturelle."""


# ══════════════════════════════════════════════════════════════════
# CHARGEMENT MODÈLE
# ══════════════════════════════════════════════════════════════════

def load_model():
    from unsloth import FastLanguageModel
    import torch

    print(f"🔵 Chargement du modèle : {MODEL_ID}")
    print(f"   GPU disponible : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM totale    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_ID,
        max_seq_length = MAX_SEQ_LEN,
        dtype          = None,   # auto-détection (bf16 sur A10G)
        load_in_4bit   = True,   # QLoRA 4-bit — divise la VRAM par 4
    )

    print("✅ Modèle chargé")
    print("🔵 Application des adapters LoRA...")

    model = FastLanguageModel.get_peft_model(
        model,
        r                   = LORA_CONFIG["r"],
        lora_alpha          = LORA_CONFIG["lora_alpha"],
        lora_dropout        = LORA_CONFIG["lora_dropout"],
        target_modules      = LORA_CONFIG["target_modules"],
        bias                = "none",
        use_gradient_checkpointing = "unsloth",  # économise 30% de VRAM
        random_state        = 42,
    )

    # Affiche le nombre de paramètres entraînables
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ LoRA configuré")
    print(f"   Paramètres total     : {total/1e6:.1f}M")
    print(f"   Paramètres entraîn.  : {trainable/1e6:.2f}M ({100*trainable/total:.2f}%)")

    return model, tokenizer


# ══════════════════════════════════════════════════════════════════
# PRÉPARATION DU DATASET
# ══════════════════════════════════════════════════════════════════

def load_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_for_unsloth(example: dict, tokenizer) -> dict:
    """
    Convertit une entrée 'messages' en texte formaté avec apply_chat_template.
    Unsloth attend un champ 'text' avec le template complet.
    """
    messages = example.get("messages", [])

    # Injecte le system prompt si absent
    if messages and messages[0]["role"] != "system":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    elif not messages:
        return {"text": ""}

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize           = False,
            add_generation_prompt = False,
        )
    except Exception:
        # Fallback Llama 3 manuel
        text = "<|begin_of_text|>"
        for msg in messages:
            role    = msg["role"]
            content = msg["content"]
            text += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

    return {"text": text}


def prepare_datasets(tokenizer):
    from datasets import Dataset

    print("\n🔵 Chargement des données...")

    train_raw = load_jsonl(DATA_DIR / "train.jsonl")
    valid_raw = load_jsonl(DATA_DIR / "valid.jsonl")

    print(f"   Train : {len(train_raw)} exemples")
    print(f"   Valid : {len(valid_raw)} exemples")

    # Conversion
    train_texts = [format_for_unsloth(ex, tokenizer)["text"] for ex in train_raw]
    valid_texts = [format_for_unsloth(ex, tokenizer)["text"] for ex in valid_raw]

    # Filtrage des entrées vides
    train_texts = [t for t in train_texts if len(t) > 20]
    valid_texts = [t for t in valid_texts if len(t) > 20]

    train_ds = Dataset.from_dict({"text": train_texts})
    valid_ds = Dataset.from_dict({"text": valid_texts})

    print(f"✅ Datasets prêts")
    print(f"   Exemple train :\n   {train_texts[0][:200]}...")

    return train_ds, valid_ds


# ══════════════════════════════════════════════════════════════════
# ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════

def train(model, tokenizer, train_ds, valid_ds, max_steps: int):
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported

    print(f"\n🔵 Lancement de l'entraînement — {max_steps} steps")
    print(f"   Batch effectif : {TRAIN_CONFIG['per_device_train_batch_size']} × {TRAIN_CONFIG['gradient_accumulation_steps']} = {TRAIN_CONFIG['per_device_train_batch_size'] * TRAIN_CONFIG['gradient_accumulation_steps']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        per_device_train_batch_size  = TRAIN_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps  = TRAIN_CONFIG["gradient_accumulation_steps"],
        warmup_steps                 = TRAIN_CONFIG["warmup_steps"],
        max_steps                    = max_steps,
        learning_rate                = TRAIN_CONFIG["learning_rate"],
        lr_scheduler_type            = TRAIN_CONFIG["lr_scheduler_type"],
        weight_decay                 = TRAIN_CONFIG["weight_decay"],
        fp16                         = not is_bfloat16_supported(),
        bf16                         = is_bfloat16_supported(),
        logging_steps                = TRAIN_CONFIG["logging_steps"],
        eval_steps                   = TRAIN_CONFIG["eval_steps"],
        save_steps                   = TRAIN_CONFIG["save_steps"],
        output_dir                   = TRAIN_CONFIG["output_dir"],
        evaluation_strategy          = TRAIN_CONFIG["evaluation_strategy"],
        save_total_limit             = TRAIN_CONFIG["save_total_limit"],
        load_best_model_at_end       = TRAIN_CONFIG["load_best_model_at_end"],
        report_to                    = TRAIN_CONFIG["report_to"],
        seed                         = 42,
    )

    trainer = SFTTrainer(
        model           = model,
        tokenizer       = tokenizer,
        train_dataset   = train_ds,
        eval_dataset    = valid_ds,
        dataset_text_field = "text",
        max_seq_length  = MAX_SEQ_LEN,
        args            = args,
    )

    # Lance l'entraînement
    print("\n" + "═"*60)
    trainer_stats = trainer.train()
    print("═"*60)

    # Stats finales
    runtime  = trainer_stats.metrics.get("train_runtime", 0)
    loss     = trainer_stats.metrics.get("train_loss", 0)
    print(f"\n✅ Entraînement terminé !")
    print(f"   Durée      : {runtime/3600:.2f}h")
    print(f"   Train loss : {loss:.4f}")

    return trainer


# ══════════════════════════════════════════════════════════════════
# SAUVEGARDE
# ══════════════════════════════════════════════════════════════════

def save_model(model, tokenizer, trainer):
    print("\n🔵 Sauvegarde du modèle...")

    # 1. Sauvegarde des adapters LoRA seuls (léger, quelques MB)
    adapter_path = OUTPUT_DIR / "adapters"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"   ✅ Adapters LoRA : {adapter_path}")

    # 2. Fusion et sauvegarde du modèle complet (4-bit quantizé)
    merged_path = OUTPUT_DIR / "merged_4bit"
    model.save_pretrained_merged(
        str(merged_path),
        tokenizer,
        save_method = "merged_4bit_forced",
    )
    print(f"   ✅ Modèle fusionné (4-bit) : {merged_path}")

    # 3. Export GGUF pour utilisation avec llama.cpp / Ollama sur Mac
    gguf_path = OUTPUT_DIR / "gguf"
    try:
        model.save_pretrained_gguf(
            str(gguf_path),
            tokenizer,
            quantization_method = "q4_k_m",  # bonne qualité, petit fichier
        )
        print(f"   ✅ GGUF (q4_k_m) : {gguf_path}  ← pour Ollama sur Mac !")
    except Exception as e:
        print(f"   ⚠️  GGUF skippé : {e}")

    print(f"\n✅ Tout sauvegardé dans : {OUTPUT_DIR}/")


# ══════════════════════════════════════════════════════════════════
# TEST RAPIDE POST-ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════

def quick_test(model, tokenizer):
    from unsloth import FastLanguageModel

    print("\n🔵 Test rapide post-entraînement...")
    FastLanguageModel.for_inference(model)

    test_prompts = [
        "Traduis en Dioula : Je mange du riz.",
        "Traduis en Dioula : Nous n'avons pas vendu le tissu.",
        "Traduis en français : Ne bɛ malo dumu.",
        "Quelle est la structure d'une phrase en Dioula ?",
        "I ni ce !",
    ]

    import torch
    for prompt in test_prompts:
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize              = True,
            add_generation_prompt = True,
            return_tensors        = "pt",
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids  = inputs,
                max_new_tokens = 80,
                temperature    = 0.1,
                do_sample      = True,
            )

        response = tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens = True,
        ).strip()

        print(f"\n  Q: {prompt}")
        print(f"  R: {response}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters",  type=int, default=20000,
                        help="Nombre de steps d'entraînement (défaut: 20000)")
    parser.add_argument("--model",  default=MODEL_ID,
                        help="Modèle HuggingFace à utiliser")
    parser.add_argument("--data",   default="data",
                        help="Dossier contenant train.jsonl et valid.jsonl")
    parser.add_argument("--no-gguf", action="store_true",
                        help="Ne pas exporter en GGUF")
    args = parser.parse_args()

    global DATA_DIR
    DATA_DIR = Path(args.data)

    print("\n" + "═"*60)
    print("  🎯 FINE-TUNING DIOULA — AWS GPU")
    print("═"*60)
    print(f"  Modèle  : {args.model}")
    print(f"  Steps   : {args.iters:,}")
    print(f"  Data    : {DATA_DIR}")
    print("═"*60 + "\n")

    # Pipeline
    model, tokenizer     = load_model()
    train_ds, valid_ds   = prepare_datasets(tokenizer)
    trainer              = train(model, tokenizer, train_ds, valid_ds, args.iters)
    quick_test(model, tokenizer)
    save_model(model, tokenizer, trainer)

    print("\n" + "═"*60)
    print("  ✅ PIPELINE COMPLET !")
    print("═"*60)
    print(f"""
  Prochaine étape — utiliser le modèle sur Mac avec Ollama :

  1. Copie le fichier GGUF depuis AWS :
     scp -i ta-cle.pem ubuntu@<ip>:{OUTPUT_DIR}/gguf/*.gguf ./

  2. Crée un Modelfile :
     echo 'FROM ./llama-dioula.gguf' > Modelfile

  3. Installe dans Ollama :
     ollama create llama-dioula -f Modelfile

  4. Discute avec ton modèle :
     ollama run llama-dioula "Traduis en Dioula : Bonjour !"
""")


if __name__ == "__main__":
    main()
