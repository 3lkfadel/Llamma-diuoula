#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Installation complète sur AWS – à lancer UNE SEULE FOIS
# Testé sur : Deep Learning AMI (Ubuntu 22.04) + CUDA 12.1
# Instance recommandée : g4dn.xlarge ou g5.xlarge
# ─────────────────────────────────────────────────────────────

set -e   # arrête si une commande échoue

echo "===== 1. Vérification CUDA ====="
nvcc --version || echo "AVERTISSEMENT: nvcc non trouvé (normal si CUDA est via les drivers)"
nvidia-smi

echo ""
echo "===== 2. Mise à jour pip ====="
pip install --upgrade pip setuptools wheel

echo ""
echo "===== 3. Installation PyTorch (CUDA 12.1) ====="
pip install torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "===== 4. Installation des dépendances ====="
pip install -r requirements_aws.txt

echo ""
echo "===== 5. Vérification finale ====="
python - <<'EOF'
import torch
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA dispo   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

import transformers, peft, trl, bitsandbytes, datasets, accelerate
print(f"transformers : {transformers.__version__}")
print(f"peft         : {peft.__version__}")
print(f"trl          : {trl.__version__}")
print(f"bitsandbytes : {bitsandbytes.__version__}")
print(f"datasets     : {datasets.__version__}")
print(f"accelerate   : {accelerate.__version__}")
print("")
print("✅ Tout est prêt – tu peux lancer le fine-tuning.")
EOF
