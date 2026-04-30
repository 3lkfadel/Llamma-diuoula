"""
prepare_dioula_dataset.py
=========================
Prépare et augmente le dataset Dioula pour fine-tuning avec mlx-lm (Llama sur Mac M1/M2/M3).

Usage:
    python prepare_dioula_dataset.py

Sorties:
    train.jsonl        → données d'entraînement (80%)
    valid.jsonl        → données de validation (10%)
    test.jsonl         → données de test (10%)
    dataset_stats.json → rapport complet sur le dataset
    system_prompt.txt  → le system prompt utilisé (pour référence)
"""

import json
import random
import os
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

ALPACA_FILE   = "dioula_alpaca_v3_finetune.json"
DATASET_FILE  = "dioula_dataset_v3_final.json"
OUTPUT_DIR    = Path(".")   # change si tu veux un autre dossier

SPLIT_TRAIN   = 0.80
SPLIT_VALID   = 0.10
SPLIT_TEST    = 0.10

RANDOM_SEED   = 42

# ─── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un assistant expert en langue Dioula (Jula), parlée principalement en Côte d'Ivoire, au Burkina Faso et au Mali.

Tes capacités :
- Traduire du français vers le Dioula et du Dioula vers le français
- Expliquer la grammaire Dioula (structure SOV, auxiliaires aspectuels, tons)
- Répondre en Dioula si l'utilisateur te parle en Dioula
- Aider à apprendre le vocabulaire Dioula

Points clés de la grammaire Dioula :
- Structure de base : Sujet + Auxiliaire + Objet + Verbe (SOV)
- Auxiliaire 'bɛ' = présent affirmatif | 'tɛ' = présent négatif
- Auxiliaire 'ye' = passé affirmatif | 'ma' = passé négatif  
- Auxiliaire 'bɛna' = futur affirmatif | 'tɛna' = futur négatif
- Langue tonale : les tons changent le sens des mots
- Voyelles spéciales : ɛ (è ouvert), ɔ (o ouvert), ŋ (ng nasal)

Réponds toujours de manière précise et naturelle."""

# ─── Chargement des données ────────────────────────────────────────────────────

def load_data():
    """Charge les deux fichiers JSON."""
    # Cherche dans le répertoire courant ou uploads
    for base in [".", "/mnt/user-data/uploads"]:
        alpaca_path = os.path.join(base, ALPACA_FILE)
        dataset_path = os.path.join(base, DATASET_FILE)
        if os.path.exists(alpaca_path) and os.path.exists(dataset_path):
            break
    else:
        raise FileNotFoundError(
            f"Fichiers introuvables. Place {ALPACA_FILE} et {DATASET_FILE} dans le même dossier que ce script."
        )

    with open(alpaca_path, encoding="utf-8") as f:
        alpaca_raw = json.load(f)

    with open(dataset_path, encoding="utf-8") as f:
        dataset_raw = json.load(f)

    print(f"✅ Chargé : {len(alpaca_raw)} entrées Alpaca")
    print(f"✅ Chargé : {len(dataset_raw['vocabulaire'])} mots de vocabulaire")
    print(f"✅ Chargé : {len(dataset_raw['grammaire'])} règles de grammaire")

    return alpaca_raw, dataset_raw

# ─── Formatage en chat (format mlx-lm) ────────────────────────────────────────

def format_as_chat(instruction: str, output: str, input_text: str = "") -> dict:
    """
    Convertit une paire instruction/output au format chat utilisé par mlx-lm.
    
    mlx-lm attend du JSONL avec des 'conversations' ou 'text' selon la config.
    On utilise le format 'text' avec le template ChatML qui est universel.
    """
    user_content = instruction
    if input_text and input_text.strip():
        user_content = f"{instruction}\n\n{input_text.strip()}"

    # Format ChatML — compatible Llama 3.x via mlx-lm
    text = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{output}<|eot_id|>"
    )

    return {"text": text}

# ─── Augmentation depuis le vocabulaire ───────────────────────────────────────

def augment_from_vocabulary(vocab: list) -> list:
    """Génère des paires d'entraînement depuis le vocabulaire."""
    pairs = []

    for entry in vocab:
        fr  = entry.get("français", "")
        dj  = entry.get("dioula", "")
        ex  = entry.get("exemple_phrase", "")
        cat = entry.get("categorie", "")
        phon = entry.get("phonétique", "")

        if not fr or not dj:
            continue

        # Paire 1 : traduction fr → dioula (mot seul)
        pairs.append({
            "instruction": f"Comment dit-on '{fr}' en Dioula ?",
            "input": "",
            "output": dj,
        })

        # Paire 2 : traduction dioula → fr
        pairs.append({
            "instruction": f"Que veut dire '{dj}' en français ?",
            "input": "",
            "output": fr,
        })

        # Paire 3 : exemple de phrase si disponible
        if ex and ex.strip() and ex.strip() != dj:
            pairs.append({
                "instruction": f"Donne un exemple de phrase avec le mot Dioula pour '{fr}'.",
                "input": "",
                "output": ex,
            })

        # Paire 4 : prononciation si disponible
        if phon and phon.strip():
            pairs.append({
                "instruction": f"Comment prononce-t-on '{dj}' ('{fr}' en Dioula) ?",
                "input": "",
                "output": f"On prononce '{dj}' comme '{phon}'.",
            })

    print(f"   → {len(pairs)} paires générées depuis le vocabulaire ({len(vocab)} mots)")
    return pairs

# ─── Augmentation depuis la grammaire ─────────────────────────────────────────

def augment_from_grammar(grammar: list) -> list:
    """Génère des paires d'entraînement depuis les règles de grammaire."""
    pairs = []

    for rule in grammar:
        regle       = rule.get("regle", "")
        description = rule.get("description", "")
        exemples    = rule.get("exemples", [])
        marqueur    = rule.get("marqueur", "")
        structure   = rule.get("structure", "")
        comparaison = rule.get("comparaison", {})

        if not regle or not description:
            continue

        # Paire 1 : explication de la règle
        reponse_regle = description
        if structure:
            reponse_regle += f"\n\nStructure : {structure}"
        if marqueur:
            reponse_regle += f"\nMarqueur : '{marqueur}'"
        if comparaison:
            fr_ex = comparaison.get("francais", "")
            dj_ex = comparaison.get("dioula", "")
            if fr_ex and dj_ex:
                reponse_regle += f"\n\nExemple :\n- Français : {fr_ex}\n- Dioula : {dj_ex}"

        pairs.append({
            "instruction": f"Explique la règle de grammaire Dioula : {regle}",
            "input": "",
            "output": reponse_regle,
        })

        # Paires depuis les exemples de la règle
        for ex in exemples[:3]:  # max 3 exemples par règle
            fr_ex = ex.get("francais", "")
            dj_ex = ex.get("dioula", "")
            glose = ex.get("glose", "")

            if fr_ex and dj_ex:
                output_ex = dj_ex
                if glose:
                    output_ex += f"\n(Glose : {glose})"

                pairs.append({
                    "instruction": f"Traduis en Dioula : {fr_ex}",
                    "input": "",
                    "output": output_ex,
                })

                pairs.append({
                    "instruction": f"Que signifie en français : '{dj_ex}' ?",
                    "input": "",
                    "output": fr_ex,
                })

    print(f"   → {len(pairs)} paires générées depuis la grammaire ({len(grammar)} règles)")
    return pairs

# ─── Paires de conversation naturelle ─────────────────────────────────────────

def add_conversation_pairs() -> list:
    """Ajoute des paires de conversation naturelle français ↔ Dioula."""

    # Conversations naturelles codées en dur (basées sur la grammaire Dioula)
    conversations = [
        # Salutations
        ("Bonjour !", "I ni ce !"),
        ("Bonjour (le matin)", "I ni sogoma"),
        ("Bonne après-midi", "I ni tile"),
        ("Bonsoir", "I ni wula"),
        ("Comment tu t'appelles ?", "I tɔgɔ di ?"),
        ("Je m'appelle Amadou.", "N tɔgɔ Amadou."),
        ("Tu viens d'où ?", "I bɔra min ?"),
        ("Je viens d'Abidjan.", "Ne bɔra Abijanu."),
        ("Au revoir !", "Ka kɛ a la !"),
        ("Merci beaucoup !", "I ni baaraka kosɛbɛ !"),
        ("S'il te plaît.", "I ni kanuya."),
        ("Pas de problème.", "Gɛlɛya si tɛ."),

        # Présent affirmatif (bɛ)
        ("Je mange du riz.", "Ne bɛ malo dumu."),
        ("Tu vas au marché.", "I bɛ taa sugu la."),
        ("Il boit de l'eau.", "A bɛ ji min."),
        ("Elle travaille.", "A bɛ baara kɛ."),
        ("Nous parlons Dioula.", "An bɛ julakan kumana."),
        ("Ils étudient.", "Olu bɛ kalan."),

        # Présent négatif (tɛ)
        ("Je ne mange pas.", "Ne tɛ dumu."),
        ("Il ne travaille pas.", "A tɛ baara kɛ."),
        ("Nous ne parlons pas français.", "An tɛ faransi kumana."),

        # Passé (ye / ma)
        ("J'ai mangé.", "Ne ye dumu."),
        ("Il est parti.", "A ye taa."),
        ("Je n'ai pas mangé.", "Ne ma dumu."),

        # Futur (bɛna / tɛna)
        ("Je vais manger.", "Ne bɛna dumu."),
        ("Il va venir.", "A bɛna na."),
        ("Je ne viendrai pas.", "Ne tɛna na."),

        # Chiffres et commerce
        ("Combien ça coûte ?", "A joli ye joli ye ?"),
        ("C'est trop cher.", "A gwa kosɛbɛ."),
        ("J'ai de l'argent.", "Ne bɛ wari la."),

        # Besoins quotidiens
        ("J'ai faim.", "Kɔngɔ bɛ ne la."),
        ("J'ai soif.", "Kɔngɔ ji bɛ ne la."),
        ("Je suis fatigué.", "Gɛlɛya bɛ ne la."),
        ("Je ne comprends pas.", "Ne tɛ a faamu."),
        ("Répète s'il te plaît.", "A fɔ kokura i ni kanuya."),
        ("Parle lentement.", "Kuma sanfɛ."),
    ]

    pairs = []
    for fr, dj in conversations:
        # fr → dioula
        pairs.append({
            "instruction": f"Traduis en Dioula : {fr}",
            "input": "",
            "output": dj,
        })
        # dioula → fr
        pairs.append({
            "instruction": f"Traduis en français : {dj}",
            "input": "",
            "output": fr,
        })

    print(f"   → {len(pairs)} paires de conversation naturelle ajoutées")
    return pairs

# ─── Pipeline principal ────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n🔵 ÉTAPE 1 — Chargement des fichiers source")
    alpaca_raw, dataset_raw = load_data()

    vocab   = dataset_raw["vocabulaire"]
    grammar = dataset_raw["grammaire"]

    # ── Rassembler toutes les paires brutes ──
    print("\n🔵 ÉTAPE 2 — Collecte et augmentation des données")

    all_pairs = []

    # Source 1 : paires Alpaca originales (169)
    for entry in alpaca_raw:
        all_pairs.append({
            "instruction": entry["instruction"],
            "input":       entry.get("input", ""),
            "output":      entry["output"],
            "source":      "alpaca_original",
        })
    print(f"   → {len(alpaca_raw)} paires Alpaca originales")

    # Source 2 : vocabulaire (210 mots → ~630 paires)
    vocab_pairs = augment_from_vocabulary(vocab)
    for p in vocab_pairs:
        p["source"] = "vocabulaire"
        all_pairs.append(p)

    # Source 3 : grammaire (24 règles → ~170 paires)
    grammar_pairs = augment_from_grammar(grammar)
    for p in grammar_pairs:
        p["source"] = "grammaire"
        all_pairs.append(p)

    # Source 4 : conversations naturelles (~70 paires)
    conv_pairs = add_conversation_pairs()
    for p in conv_pairs:
        p["source"] = "conversation"
        all_pairs.append(p)

    print(f"\n   📊 Total brut : {len(all_pairs)} paires")

    # ── Dédoublonnage sur (instruction, output) ──
    print("\n🔵 ÉTAPE 3 — Dédoublonnage")
    seen = set()
    unique_pairs = []
    dupes = 0
    for p in all_pairs:
        key = (p["instruction"].strip().lower(), p["output"].strip().lower())
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)
        else:
            dupes += 1

    print(f"   → {dupes} doublons supprimés")
    print(f"   → {len(unique_pairs)} paires uniques conservées")

    # ── Formatage en ChatML ──
    print("\n🔵 ÉTAPE 4 — Formatage en ChatML (format mlx-lm)")
    formatted = []
    for p in unique_pairs:
        formatted.append(format_as_chat(
            instruction=p["instruction"],
            output=p["output"],
            input_text=p.get("input", ""),
        ))

    # ── Split train / valid / test ──
    print("\n🔵 ÉTAPE 5 — Split train / valid / test")
    random.shuffle(formatted)
    n = len(formatted)
    n_train = int(n * SPLIT_TRAIN)
    n_valid = int(n * SPLIT_VALID)

    train_data = formatted[:n_train]
    valid_data = formatted[n_train:n_train + n_valid]
    test_data  = formatted[n_train + n_valid:]

    print(f"   → Train : {len(train_data)} exemples ({SPLIT_TRAIN*100:.0f}%)")
    print(f"   → Valid : {len(valid_data)} exemples ({SPLIT_VALID*100:.0f}%)")
    print(f"   → Test  : {len(test_data)}  exemples ({SPLIT_TEST*100:.0f}%)")

    # ── Écriture des fichiers JSONL ──
    print("\n🔵 ÉTAPE 6 — Écriture des fichiers JSONL")

    def write_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"   ✅ {path} ({len(data)} lignes)")

    write_jsonl(train_data, OUTPUT_DIR / "train.jsonl")
    write_jsonl(valid_data, OUTPUT_DIR / "valid.jsonl")
    write_jsonl(test_data,  OUTPUT_DIR / "test.jsonl")

    # ── Sauvegarde du system prompt ──
    prompt_path = OUTPUT_DIR / "system_prompt.txt"
    prompt_path.write_text(SYSTEM_PROMPT, encoding="utf-8")
    print(f"   ✅ {prompt_path}")

    # ── Rapport stats ──
    from collections import Counter
    sources = Counter(p.get("source", "?") for p in unique_pairs)

    stats = {
        "total_paires_brutes":    len(all_pairs),
        "doublons_supprimes":     dupes,
        "total_paires_uniques":   len(unique_pairs),
        "split": {
            "train": len(train_data),
            "valid": len(valid_data),
            "test":  len(test_data),
        },
        "par_source": dict(sources),
        "format": "ChatML (Llama 3.x)",
        "modele_cible": "meta-llama/Llama-3.2-3B-Instruct",
        "framework": "mlx-lm",
    }

    stats_path = OUTPUT_DIR / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"   ✅ {stats_path}")

    # ── Aperçu d'un exemple formaté ──
    print("\n🔵 APERÇU — 1 exemple formaté (train.jsonl)")
    print("─" * 60)
    sample = train_data[0]["text"]
    # Affiche de manière lisible
    sample_readable = (
        sample
        .replace("<|begin_of_text|>", "[BEGIN]\n")
        .replace("<|start_header_id|>", "\n[")
        .replace("<|end_header_id|>", "]\n")
        .replace("<|eot_id|>", "\n[EOT]\n")
    )
    print(sample_readable)

    print("\n" + "═" * 60)
    print("✅ DATASET PRÊT !")
    print("═" * 60)
    print()
    print("📁 Fichiers générés :")
    print("   train.jsonl, valid.jsonl, test.jsonl, dataset_stats.json, system_prompt.txt")
    print()
    print("🚀 PROCHAINES ÉTAPES pour fine-tuner sur Mac M1 Pro :")
    print()
    print("  1. Installer mlx-lm :")
    print("     pip install mlx-lm")
    print()
    print("  2. Télécharger le modèle (compte Hugging Face requis) :")
    print("     mlx_lm.convert --hf-path meta-llama/Llama-3.2-3B-Instruct \\")
    print("         --mlx-path ./llama-3.2-3b-mlx -q")
    print()
    print("  3. Lancer le fine-tuning :")
    print("     mlx_lm.lora \\")
    print("         --model ./llama-3.2-3b-mlx \\")
    print("         --train \\")
    print("         --data . \\")
    print("         --num-layers 8 \\")
    print("         --iters 800 \\")
    print("         --batch-size 2 \\")
    print("         --learning-rate 1e-4 \\")
    print("         --val-batches 10")
    print()
    print("  4. Tester le modèle :")
    print("     mlx_lm.generate \\")
    print("         --model ./llama-3.2-3b-mlx \\")
    print("         --adapter-path ./adapters \\")
    print('         --prompt "Traduis en Dioula : Bonjour, comment vas-tu ?"')
    print()
    print("  5. (Optionnel) Fusionner les adapters dans le modèle :")
    print("     mlx_lm.fuse \\")
    print("         --model ./llama-3.2-3b-mlx \\")
    print("         --adapter-path ./adapters \\")
    print("         --save-path ./llama-dioula-final")


if __name__ == "__main__":
    main()
