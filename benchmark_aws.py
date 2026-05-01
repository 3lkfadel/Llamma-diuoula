#!/usr/bin/env python3
"""
Benchmark Dioula — version AWS/CUDA
Même tests que benchmark_dioula.py mais avec transformers au lieu de mlx_lm
"""

import json, time, argparse
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Cas de test (identiques à benchmark_dioula.py) ────────────
BENCHMARK_CASES = {
    "baseline_vu": {"description":"Phrases proches du dataset","weight":1.0,"cases":[
        {"prompt":"Traduis en Dioula : Bonjour, comment vas-tu ?","expected":"I ni sogoma, i ka kɛnɛ wa ?","keywords":["ni","kɛnɛ","wa"]},
        {"prompt":"Traduis en Dioula : Je mange du riz.","expected":"Ne bɛ malo dumu.","keywords":["bɛ","malo","dumu"]},
        {"prompt":"Traduis en Dioula : Je ne travaille pas.","expected":"Ne tɛ baara kɛ.","keywords":["tɛ","baara"]},
        {"prompt":"Traduis en Dioula : Merci beaucoup.","expected":"I ni baaraka kosɛbɛ.","keywords":["baaraka"]},
    ]},
    "auxiliaires": {"description":"Les 6 auxiliaires","weight":2.0,"cases":[
        {"prompt":"Traduis en Dioula : Tu travailles.","expected":"I bɛ baara kɛ.","keywords":["bɛ","baara"]},
        {"prompt":"Traduis en Dioula : Tu ne travailles pas.","expected":"I tɛ baara kɛ.","keywords":["tɛ","baara"]},
        {"prompt":"Traduis en Dioula : Tu as travaillé.","expected":"I ye baara kɛ.","keywords":["ye","baara"]},
        {"prompt":"Traduis en Dioula : Tu n'as pas travaillé.","expected":"I ma baara kɛ.","keywords":["ma","baara"]},
        {"prompt":"Traduis en Dioula : Tu vas travailler.","expected":"I bɛna baara kɛ.","keywords":["bɛna","baara"]},
        {"prompt":"Traduis en Dioula : Tu ne vas pas travailler.","expected":"I tɛna baara kɛ.","keywords":["tɛna","baara"]},
    ]},
    "sov_nouveaux": {"description":"SOV phrases jamais vues","weight":2.0,"cases":[
        {"prompt":"Traduis en Dioula : Nous n'avons pas vendu le tissu.","expected":"An ma fini feere.","keywords":["ma","fini","feere"]},
        {"prompt":"Traduis en Dioula : Ils vont boire du lait demain.","expected":"U bɛna nɔnɔ min sini.","keywords":["bɛna","nɔnɔ","min"]},
        {"prompt":"Traduis en Dioula : Vous ne lirez pas le journal.","expected":"Aw tɛna jurnali kalan.","keywords":["tɛna","jurnali","kalan"]},
        {"prompt":"Traduis en Dioula : Elle a acheté des chaussures.","expected":"A ye daaw san.","keywords":["ye","daaw","san"]},
        {"prompt":"Traduis en Dioula : Tu cherches de l'argent.","expected":"I bɛ wari ɲini.","keywords":["bɛ","wari","ɲini"]},
    ]},
    "possession": {"description":"Possession avec ka","weight":1.5,"cases":[
        {"prompt":"Comment dit-on 'ma maison' en Dioula ?","expected":"ne ka so","keywords":["ka","so"]},
        {"prompt":"Comment dit-on 'leur voiture' en Dioula ?","expected":"u ka mobili","keywords":["ka","mobili"]},
        {"prompt":"Comment dit-on 'ton père' en Dioula ?","expected":"i ka fa","keywords":["ka","fa"]},
    ]},
    "questions": {"description":"Questions","weight":1.5,"cases":[
        {"prompt":"Traduis en Dioula : Est-ce que tu manges ?","expected":"I bɛ dumu wa ?","keywords":["dumu","wa"]},
        {"prompt":"Traduis en Dioula : Que fais-tu ?","expected":"I bɛ mun kɛ ?","keywords":["mun","kɛ"]},
        {"prompt":"Traduis en Dioula : Où est le marché ?","expected":"Sugu bɛ yɔrɔ di ?","keywords":["yɔrɔ","di"]},
        {"prompt":"Traduis en Dioula : Quel est ton nom ?","expected":"I tɔgɔ di ?","keywords":["tɔgɔ","di"]},
    ]},
    "dioula_vers_fr": {"description":"Dioula vers Français","weight":1.5,"cases":[
        {"prompt":"Traduis en français : Ne bɛ malo dumu.","expected":"Je mange du riz.","keywords":["mange","riz"]},
        {"prompt":"Traduis en français : A tɛ baara kɛ.","expected":"Il ne travaille pas.","keywords":["travaille","pas"]},
        {"prompt":"Traduis en français : An bɛna sugu la taa.","expected":"Nous allons au marché.","keywords":["marché","allons"]},
    ]},
    "grammaire": {"description":"Explication grammaire","weight":1.0,"cases":[
        {"prompt":"Quelle est la structure de base d'une phrase en Dioula ?","expected":"SOV","keywords":["SOV","Verbe"]},
        {"prompt":"C'est quoi l'auxiliaire bɛ en Dioula ?","expected":"présent affirmatif","keywords":["présent"]},
        {"prompt":"Comment dit-on il et elle en Dioula ?","expected":"a","keywords":["genre"]},
    ]},
}

SYSTEM = "Tu es un assistant expert en langue Dioula. Tu traduis entre le français et le Dioula. Réponds uniquement avec la traduction ou l'explication."


def build_prompt(tokenizer, text):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )


def generate(model, tokenizer, prompt_text, max_new_tokens=80):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    # On ne garde que les tokens générés (pas le prompt)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    resp = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return resp.strip()


def score(resp, kws):
    r = resp.lower()
    found = [k for k in kws if k.lower() in r]
    return len(found) / len(kws) if kws else 0.0, found


def bar(s, w=28):
    f = int(s * w)
    return f"[{'█'*f}{'░'*(w-f)}] {s*100:.0f}%"


def c(t, code):
    return f"\033[{code}m{t}\033[0m"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                   help="Modèle de base HuggingFace ou chemin local")
    p.add_argument("--adapter",    default="./adapters_aws/final",
                   help="Chemin vers les adaptateurs LoRA (dossier final)")
    p.add_argument("--no-adapter", action="store_true",
                   help="Tester le modèle de base sans adaptateurs")
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--hf-token",   default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponible — GPU requis.")

    hf_kwargs = {}
    if args.hf_token:
        hf_kwargs["token"] = args.hf_token

    print("\n" + "═"*65)
    print("  🎯 BENCHMARK DIOULA (AWS)")
    print("═"*65)
    print(f"  Modèle  : {args.model}")
    print(f"  Adapter : {'non' if args.no_adapter else args.adapter}")
    print(f"  Date    : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═"*65)

    print("\n⏳ Chargement du modèle...")
    t0 = time.time()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, **hf_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        **hf_kwargs,
    )

    if not args.no_adapter:
        # On NE merge PAS : merger LoRA dans un modèle 4-bit corrompt les poids
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()
    print(f"✅ Chargé en {time.time()-t0:.1f}s\n")

    results = {}
    tw = 0.0
    tt = 0.0

    for cid, cat in BENCHMARK_CASES.items():
        desc = cat["description"]
        w = cat["weight"]
        cases = cat["cases"]
        print(f"\n{'─'*65}\n  📂 {desc.upper()}\n{'─'*65}")
        scores = []
        details = []

        for i, case in enumerate(cases, 1):
            prompt   = case["prompt"]
            expected = case["expected"]
            kws      = case["keywords"]
            print(f"\n  [{i}/{len(cases)}] {prompt}")
            fp = build_prompt(tokenizer, prompt)
            t1 = time.time()
            try:
                resp = generate(model, tokenizer, fp, max_new_tokens=args.max_tokens)
            except Exception as e:
                resp = f"[ERREUR:{e}]"
            el = time.time() - t1
            sc, found = score(resp, kws)
            scores.append(sc)
            cc = "92" if sc >= 0.8 else "93" if sc >= 0.5 else "91"
            print(f"  ✦ Réponse  : {c(resp[:120], '97')}")
            print(f"  ✦ Attendu  : {c(expected[:120], '90')}")
            print(f"  ✦ Mots-clés: {c(str(found), '96')} / {kws}")
            print(f"  ✦ Score    : {c(f'{sc*100:.0f}%', cc)}  ({el:.1f}s)")
            details.append({"prompt": prompt, "expected": expected,
                             "response": resp, "score": round(sc, 3), "found": found})

        avg = sum(scores) / len(scores) if scores else 0.0
        tw += avg * w
        tt += w
        bc = "92" if avg >= 0.8 else "93" if avg >= 0.5 else "91"
        print(f"\n  Catégorie : {c(bar(avg), bc)}  (poids ×{w})")
        results[cid] = {"description": desc, "weight": w,
                        "score": round(avg, 3), "details": details}

    gs = tw / tt if tt else 0.0
    print("\n" + "═"*65 + "\n  📊 RÉSULTATS GLOBAUX\n" + "═"*65)
    for cid, r in results.items():
        s = r["score"]
        icon = "✅" if s >= 0.8 else "⚠️ " if s >= 0.5 else "❌"
        cc   = "92" if s >= 0.8 else "93" if s >= 0.5 else "91"
        print(f"  {icon}  {r['description']:<44} {c(f'{s*100:.0f}%', cc)}")
    print("─"*65)
    gc = "92" if gs >= 0.8 else "93" if gs >= 0.5 else "91"
    print(f"\n  🏆 SCORE GLOBAL : {c(f'{gs*100:.1f}%', gc)}")
    print(f"  {c(bar(gs, 48), gc)}")
    print("\n  Interprétation :")
    if gs >= 0.85:   print(c("  → Excellent ! Bonne généralisation grammaticale.", "92"))
    elif gs >= 0.65: print(c("  → Bon. Relancer avec --iters 1500 pour affiner.", "93"))
    elif gs >= 0.45: print(c("  → Moyen. Augmenter le dataset.", "93"))
    else:            print(c("  → Faible. Vérifier format données et réentraîner.", "91"))

    out = Path("benchmark_report.json")
    out.write_text(json.dumps({
        "date": datetime.now().isoformat(),
        "model": args.model,
        "global_score": round(gs, 3),
        "categories": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  💾 Rapport : {out}\n" + "═"*65 + "\n")


if __name__ == "__main__":
    main()
