# 🌍 Llama Dioula — Fine-tuning Llama 3.2 pour la langue Dioula (Jula)

> Fine-tuning d'un modèle Llama 3.2 pour comprendre et traduire le **Dioula (Jula)**, langue parlée par plus de 12 millions de personnes en Côte d'Ivoire, au Burkina Faso et au Mali.

---

## 🎯 Objectif

Créer un modèle capable de :
- Traduire **Français → Dioula** et **Dioula → Français**
- Respecter la **grammaire Dioula** (structure SOV, auxiliaires aspectuels, tons)
- Répondre en Dioula si on lui parle en Dioula
- Expliquer les règles grammaticales du Dioula

---

## 📁 Structure du projet

```
Llamma-diuoula/
│
├── data/                          # Datasets d'entraînement (format JSONL)
│   ├── train.jsonl                # 80% — 4 588 exemples
│   ├── valid.jsonl                # 10% — 573 exemples
│   └── test.jsonl                 # 10% — 575 exemples
│
├── dioula_alpaca_v3_finetune.json # Dataset source Alpaca (169 paires)
├── dioula_dataset_v3_final.json   # Dataset structuré (grammaire + vocabulaire)
│
├── generate_grammar_pairs.py      # Génère le dataset depuis les règles grammaticales
├── prepare_dioula_dataset.py      # Pipeline de préparation des données (v1)
├── finetune_aws.py                # Fine-tuning sur AWS GPU avec Unsloth
├── benchmark_dioula.py            # Évaluation complète du modèle
│
└── README.md
```

---

## 📊 Dataset

Le dataset a été **construit et augmenté** depuis deux sources principales :

| Source | Paires | Description |
|--------|--------|-------------|
| Alpaca original | 169 | Paires français ↔ Dioula validées |
| Génération SOV | 5 376 | Combinatoire grammaticale (tous temps × pronoms × verbes × objets) |
| Possession (`ka`) | 100 | Structure possesseur + ka + objet |
| Questions | 54 | wa, mun, jɔn, yɔrɔ di, waati di... |
| Copule `ye` | 25 | Structure X ye Y ye |
| Q&A grammaticale | 14 | Explications des règles |
| **Total** | **5 736** | **paires uniques** |

### Format des données

Format `messages` (compatible Llama 3.x, Unsloth, mlx-lm) :

```json
{
  "messages": [
    {"role": "system", "content": "Tu es un assistant expert en langue Dioula..."},
    {"role": "user", "content": "Traduis en Dioula : Je mange du riz."},
    {"role": "assistant", "content": "Ne bɛ malo dumu."}
  ]
}
```

---

## 🧠 Grammaire Dioula apprise

Le modèle apprend les règles grammaticales suivantes :

### Structure SOV
```
Français (SVO) : Je   mange  du riz
Dioula   (SOV) : Ne   bɛ     malo    dumu
                 Suj  Aux    Objet   Verbe
```

### Les 6 auxiliaires aspectuels
| Auxiliaire | Temps | Exemple |
|-----------|-------|---------|
| `bɛ` | Présent affirmatif | Ne bɛ malo dumu = Je mange du riz |
| `tɛ` | Présent négatif | Ne tɛ dumu = Je ne mange pas |
| `ye` | Passé affirmatif | Ne ye dumu = J'ai mangé |
| `ma` | Passé négatif | Ne ma dumu = Je n'ai pas mangé |
| `bɛna` | Futur affirmatif | Ne bɛna dumu = Je vais manger |
| `tɛna` | Futur négatif | Ne tɛna dumu = Je ne vais pas manger |

### Pronoms (pas de genre)
| Français | Dioula |
|----------|--------|
| Je | Ne / N |
| Tu | I |
| Il / Elle | **A** (pas de distinction !) |
| Nous | An |
| Vous | Aw |
| Ils / Elles | U |

---

## 🚀 Installation & Usage

### Sur Mac M1/M2/M3 (mlx-lm)

```bash
# 1. Cloner le repo
git clone https://github.com/3lkfadel/Llamma-diuoula.git
cd Llamma-diuoula

# 2. Créer un environnement virtuel
python -m venv env
source env/bin/activate

# 3. Installer les dépendances
pip install mlx-lm

# 4. Générer le dataset
python generate_grammar_pairs.py

# 5. Télécharger le modèle (compte HuggingFace requis)
mlx_lm.convert --hf-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --mlx-path ./llama-3.2-3b-mlx -q

# 6. Lancer le fine-tuning
mlx_lm.lora \
    --model ./llama-3.2-3b-mlx \
    --train \
    --data "./data" \
    --num-layers 16 \
    --iters 3000 \
    --batch-size 4 \
    --learning-rate 2e-4
```

### Sur AWS GPU — Recommandé

```bash
# Instance recommandée : g5.xlarge (A10G 24 Go) — requis pour 8B en QLoRA
# AMI : Deep Learning AMI Ubuntu 22.04

# 1. Cloner le repo
git clone https://github.com/3lkfadel/Llamma-diuoula.git
cd Llamma-diuoula

# 2. Installer TOUTES les dépendances (PyTorch + CUDA + HuggingFace)
chmod +x setup_aws.sh
./setup_aws.sh

# 3. Token HuggingFace (requis pour Llama 3.2 gated)
#    → Créer un token sur https://huggingface.co/settings/tokens
#    → Accepter la licence sur https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# 4. Lancer le fine-tuning
python train_aws.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --train train.jsonl \
  --valid valid.jsonl \
  --output ./adapters_aws \
  --iters 800 \
  --hf-token $HF_TOKEN

# Pour un meilleur score (recommandé) :
python train_aws.py --iters 1500 --hf-token $HF_TOKEN

# 5. Benchmark après fine-tuning
python benchmark_aws.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --adapter ./adapters_aws/final \
  --hf-token $HF_TOKEN
```

---

## 📈 Évaluation

```bash
# Benchmark complet (9 catégories)
python benchmark_dioula.py \
    --model ./llama-3.2-3b-mlx \
    --adapter ./adapters
```

### Catégories testées
- Phrases proches du dataset (baseline)
- Structure SOV sur phrases jamais vues (généralisation)
- Maîtrise des 6 auxiliaires
- Possession avec `ka`
- Questions (wa, mun, jɔn, yɔrɔ di...)
- Copule `ye` (être / identification)
- Traduction Dioula → Français
- Explication des règles grammaticales
- Conversation naturelle

---

## 🗺️ Roadmap

- [x] Dataset de base (169 paires Alpaca)
- [x] Augmentation grammaticale (5 736 paires)
- [x] Fine-tuning sur Mac M1 Pro (mlx-lm)
- [x] Pipeline de benchmark automatique
- [ ] Fine-tuning sur AWS GPU (20 000+ steps)
- [ ] Export GGUF pour Ollama
- [ ] Interface chat web (Gradio / Streamlit)
- [ ] Dataset étendu (locuteurs natifs)
- [ ] Support Bambara et Malinké (langues apparentées)

---

## 🌐 Langues apparentées

Le Dioula fait partie de la famille **Manding** avec :
- **Bambara** (Mali) — très proche, ~80% de vocabulaire commun
- **Malinké** (Guinée, Sénégal)
- **Mandinka** (Gambie)

Un modèle Dioula pourra potentiellement être adapté à ces langues avec peu d'effort supplémentaire.

---

## 🤝 Contribuer

Les contributions sont les bienvenues, surtout :
- **Corrections linguistiques** par des locuteurs natifs
- Nouvelles paires de traduction
- Données audio pour la prononciation
- Tests sur d'autres dialectes

---

## 📄 Licence

MIT — libre d'utilisation, de modification et de distribution.

---

## ✊ Pourquoi ce projet ?

Le Dioula est une langue de commerce et de communication inter-ethnique en Afrique de l'Ouest, mais elle est **presque absente** des grands modèles de langage. Ce projet vise à changer ça, en construisant une base open-source pour la NLP en Dioula.

> *"I ni ce"* — Bonjour en Dioula 🙏
