"""
generate_grammar_pairs.py
=========================
Génère des milliers de phrases Dioula en appliquant mécaniquement
les vraies règles grammaticales (SOV, auxiliaires, possession, questions...).

Le modèle fine-tuné apprendra ainsi la LOGIQUE derrière la grammaire,
pas juste des traductions mémorisées.

Usage:
    python generate_grammar_pairs.py

Sortie:
    data/train.jsonl   (80%)
    data/valid.jsonl   (10%)
    data/test.jsonl    (10%)
"""

import json
import random
import itertools
from pathlib import Path

RANDOM_SEED = 42
OUTPUT_DIR  = Path("data")

# ══════════════════════════════════════════════════════════════════
# DICTIONNAIRE GRAMMATICAL DIOULA
# Source : règles G001-G012 du dataset
# ══════════════════════════════════════════════════════════════════

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

# ── Pronoms (G011) ──────────────────────────────────────────────
# (fr_sujet, fr_verbe_accord, dioula)
PRONOMS = [
    ("Je",    "je",    "Ne"),
    ("Tu",    "tu",    "I"),
    ("Il",    "il",    "A"),
    ("Elle",  "elle",  "A"),   # A = il/elle en dioula (G011: pas de genre)
    ("Nous",  "nous",  "An"),
    ("Vous",  "vous",  "Aw"),
    ("Ils",   "ils",   "U"),
    ("Elles", "elles", "U"),
]

# ── Auxiliaires aspectuels (G002-G005) ──────────────────────────
# (nom, auxiliaire_dioula, construction_fr_affirmative, construction_fr_negative, negatif)
AUXILIAIRES = [
    # nom,          dioula,   fr_positif,     fr_negatif,          is_neg
    ("présent",     "bɛ",    "{v}",           "ne {v} pas",        False),
    ("prés_nég",    "tɛ",    "ne {v} pas",    None,                True),
    ("passé",       "ye",    "a {v}",         "n'a pas {v}",       False),
    ("passé_nég",   "ma",    "n'a pas {v}",   None,                True),
    ("futur",       "bɛna",  "va {v}",        "ne va pas {v}",     False),
    ("futur_nég",   "tɛna",  "ne va pas {v}", None,                True),
]

# ── Verbes intransitifs (pas d'objet direct) ────────────────────
# (infinitif_fr, radical_conjugué_fr, dioula)
VERBES_INTRANS = [
    ("partir",      "pars/part/partons/partez/partent",  "taa"),
    ("venir",       "viens/vient/venons/venez/viennent", "na"),
    ("dormir",      "dors/dort/dormons/dormez/dorment",  "sɔrɔ"),
    ("courir",      "cours/court/courons/courez/courent","cɛ"),
    ("travailler",  "travaille",                          "baara kɛ"),
    ("étudier",     "étudie",                             "kalan"),
    ("parler",      "parle",                              "kuma"),
    ("chanter",     "chante",                             "donkili kɛ"),
    ("danser",      "danse",                              "fɛn kɛ"),
    ("jouer",       "joue",                               "la kɛ"),
    ("rire",        "rit",                                "nɔgɔma"),
    ("pleurer",     "pleure",                             "ɲɔgɔn"),
    ("prier",       "prie",                               "seli kɛ"),
    ("voyager",     "voyage",                             "taabolo kɛ"),
]

# ── Verbes transitifs + objets possibles ───────────────────────
# (infinitif_fr, radical_fr, dioula, [objets compatibles])
VERBES_TRANS = [
    ("manger",    "mange",   "dumu",      "NOURRITURE"),
    ("boire",     "boit",    "min",       "BOISSON"),
    ("acheter",   "achète",  "san",       "MARCHANDISE"),
    ("vendre",    "vend",    "feere",     "MARCHANDISE"),
    ("chercher",  "cherche", "ɲini",      "OBJET"),
    ("trouver",   "trouve",  "sɔrɔ",      "OBJET"),
    ("voir",      "voit",    "ye",        "OBJET"),
    ("prendre",   "prend",   "ta",        "OBJET"),
    ("porter",    "porte",   "taa ni ... ye", "OBJET"),
    ("cuisiner",  "cuisine", "na tobi",   "NOURRITURE"),
    ("lire",      "lit",     "kalan",     "TEXTE"),
    ("écrire",    "écrit",   "sɛbɛn",     "TEXTE"),
    ("aimer",     "aime",    "kanu",      "PERSONNE"),
    ("appeler",   "appelle", "wele",      "PERSONNE"),
]

# ── Objets par catégorie ────────────────────────────────────────
OBJETS = {
    "NOURRITURE": [
        ("du riz",          "malo"),
        ("du pain",         "bururu"),
        ("du poisson",      "jɛni"),
        ("de la viande",    "sogo"),
        ("des mangues",     "mangoro"),
        ("des bananes",     "namasa"),
        ("du maïs",         "kaba"),
        ("du fonio",        "foni"),
        ("du manioc",       "barakɔnɔ"),
    ],
    "BOISSON": [
        ("de l'eau",        "ji"),
        ("du café",         "kafe"),
        ("du thé",          "tii"),
        ("du lait",         "nɔnɔ"),
        ("du jus",          "dege"),
        ("du bissap",       "dabileni"),
    ],
    "MARCHANDISE": [
        ("du tissu",        "fini"),
        ("des chaussures",  "daaw"),
        ("du savon",        "sabun"),
        ("du riz",          "malo"),
        ("du sucre",        "sukaro"),
        ("de l'huile",      "tutu"),
        ("un téléphone",    "portabulu"),
    ],
    "OBJET": [
        ("son sac",         "a ka jaabi"),
        ("de l'argent",     "wari"),
        ("la voiture",      "mobili"),
        ("le livre",        "sɛbɛnni"),
        ("la clé",          "clee"),
    ],
    "TEXTE": [
        ("le Coran",        "Kuran"),
        ("le livre",        "sɛbɛnni"),
        ("le journal",      "jurnali"),
        ("la lettre",       "sɛbɛn"),
    ],
    "PERSONNE": [
        ("sa mère",         "a ka ba"),
        ("son ami",         "a ka jɔ"),
        ("son père",        "a ka fa"),
        ("le maître",       "karamɔgɔ"),
    ],
}

# ── Lieux avec postpositions (G006) ─────────────────────────────
LIEUX = [
    ("au marché",           "sugu la"),
    ("à la maison",         "so la"),
    ("à l'école",           "kalanso la"),
    ("au champ",            "farow la"),
    ("à la mosquée",        "misiri la"),
    ("en ville",            "dugu la"),
    ("au travail",          "baarakɛyɔrɔ la"),
    ("chez moi",            "ne fɛ"),
    ("chez toi",            "i fɛ"),
    ("chez lui/elle",       "a fɛ"),
]

# ── Possession (G012) ───────────────────────────────────────────
POSSESSION = [
    # (fr_possesseur, dioula_possesseur, fr_objet, dioula_objet)
    ("ma",    "ne ka", "maison",    "so"),
    ("ton",   "i ka",  "père",      "fa"),
    ("sa",    "a ka",  "mère",      "ba"),
    ("notre", "an ka", "école",     "kalanso"),
    ("votre", "aw ka", "voiture",   "mobili"),
    ("leur",  "u ka",  "argent",    "wari"),
    ("ma",    "ne ka", "moto",      "mɔtɔ"),
    ("ton",   "i ka",  "téléphone", "portabulu"),
    ("sa",    "a ka",  "ami",       "jɔ"),
    ("notre", "an ka", "village",   "dugu"),
]


# ══════════════════════════════════════════════════════════════════
# GÉNÉRATEURS DE PAIRES
# ══════════════════════════════════════════════════════════════════

def conjugue_fr(pronom_fr, verbe_radical, auxiliaire_nom):
    """Construit la phrase française selon le pronom et l'auxiliaire."""
    p = pronom_fr

    # Conjugaisons simples selon personne
    conj = {
        "Je":    {"prs": "mange/bois/travaille/pars/viens/dors/cours/étudie/parle/chante/danse/joue/rit/pleure/prie/voyage/achète/vends/cherche/trouve/vois/prends/porte/cuisine/lis/écris/aime/appelle",
                  "pss": "suis allé(e)/ai", "fut": "vais"},
        "Tu":    {"prs": "es/fais/vas", "pss": "es allé(e)/as", "fut": "vas"},
        "Il":    {"prs": "", "pss": "est allé/a", "fut": "va"},
        "Elle":  {"prs": "", "pss": "est allée/a", "fut": "va"},
        "Nous":  {"prs": "", "pss": "sommes allés/avons", "fut": "allons"},
        "Vous":  {"prs": "", "pss": "êtes allés/avez", "fut": "allez"},
        "Ils":   {"prs": "", "pss": "sont allés/ont", "fut": "vont"},
        "Elles": {"prs": "", "pss": "sont allées/ont", "fut": "vont"},
    }
    return verbe_radical  # On garde simple pour la lisibilité

def fr_phrase(pronom, verbe_fr, objet_fr, aux_nom, negatif):
    """Construit une phrase française naturelle."""
    if aux_nom == "présent":
        phrase = f"{pronom} {verbe_fr}"
        if objet_fr:
            phrase += f" {objet_fr}"
        if negatif:
            # Insert ne...pas
            phrase = f"{pronom} ne {verbe_fr} pas"
            if objet_fr:
                phrase += f" {objet_fr}"
    elif aux_nom in ("prés_nég",):
        phrase = f"{pronom} ne {verbe_fr} pas"
        if objet_fr:
            phrase += f" {objet_fr}"
    elif aux_nom == "passé":
        phrase = f"{pronom} a {verbe_fr}"
        if objet_fr:
            phrase += f" {objet_fr}"
    elif aux_nom == "passé_nég":
        phrase = f"{pronom} n'a pas {verbe_fr}"
        if objet_fr:
            phrase += f" {objet_fr}"
    elif aux_nom == "futur":
        phrase = f"{pronom} va {verbe_fr}"
        if objet_fr:
            phrase += f" {objet_fr}"
    elif aux_nom == "futur_nég":
        phrase = f"{pronom} ne va pas {verbe_fr}"
        if objet_fr:
            phrase += f" {objet_fr}"
    else:
        phrase = f"{pronom} {verbe_fr}"
    return phrase.strip() + "."

def dioula_phrase(pronom_dj, aux_dj, objet_dj, verbe_dj):
    """Construit une phrase Dioula en structure SOV : Sujet + Aux + Objet + Verbe."""
    if objet_dj:
        return f"{pronom_dj} {aux_dj} {objet_dj} {verbe_dj}."
    else:
        return f"{pronom_dj} {aux_dj} {verbe_dj}."


def generer_phrases_sov() -> list:
    """Génère toutes les combinaisons Sujet + Auxiliaire + Objet + Verbe (SOV)."""
    pairs = []

    # ── Verbes intransitifs (sans objet) ──
    for (pr_fr, _, pr_dj) in PRONOMS:
        for (inf_fr, rad_fr, vb_dj) in VERBES_INTRANS:
            for (aux_nom, aux_dj, _, _, is_neg) in AUXILIAIRES:

                fr = fr_phrase(pr_fr, rad_fr, None, aux_nom, is_neg)
                dj = dioula_phrase(pr_dj, aux_dj, None, vb_dj)

                # fr → dioula
                pairs.append({
                    "instruction": f"Traduis en Dioula : {fr}",
                    "output": dj,
                    "source": "sov_intrans",
                })
                # dioula → fr
                pairs.append({
                    "instruction": f"Traduis en français : {dj}",
                    "output": fr,
                    "source": "sov_intrans",
                })

    # ── Verbes transitifs (avec objet) ──
    for (pr_fr, _, pr_dj) in PRONOMS:
        for (inf_fr, rad_fr, vb_dj, categorie) in VERBES_TRANS:
            # Prend max 3 objets par catégorie pour éviter explosion
            objets_cat = OBJETS.get(categorie, [])[:3]
            for (obj_fr, obj_dj) in objets_cat:
                for (aux_nom, aux_dj, _, _, is_neg) in AUXILIAIRES:

                    fr = fr_phrase(pr_fr, rad_fr, obj_fr, aux_nom, is_neg)
                    dj = dioula_phrase(pr_dj, aux_dj, obj_dj, vb_dj)

                    pairs.append({
                        "instruction": f"Traduis en Dioula : {fr}",
                        "output": dj,
                        "source": "sov_trans",
                    })
                    pairs.append({
                        "instruction": f"Traduis en français : {dj}",
                        "output": fr,
                        "source": "sov_trans",
                    })

    print(f"   → {len(pairs)} paires SOV générées")
    return pairs


def generer_possession() -> list:
    """Génère des phrases de possession avec 'ka' (G012)."""
    pairs = []

    for (poss_fr, poss_dj, obj_fr, obj_dj) in POSSESSION:
        # Phrase simple de possession
        fr = f"{poss_fr} {obj_fr}"
        dj = f"{poss_dj} {obj_dj}"
        pairs.append({
            "instruction": f"Comment dit-on '{fr}' en Dioula ?",
            "output": dj,
            "source": "possession",
        })
        pairs.append({
            "instruction": f"Que signifie '{dj}' en français ?",
            "output": fr,
            "source": "possession",
        })

        # Phrases avec possession + verbe
        for (pr_fr, _, pr_dj) in PRONOMS[:4]:  # Je, Tu, Il, Nous
            for (aux_nom, aux_dj, _, _, is_neg) in AUXILIAIRES[:2]:  # présent + présent nég
                fr_p = f"{pr_fr} {'vois' if pr_fr == 'Je' else 'voit'} {poss_fr} {obj_fr}."
                dj_p = f"{pr_dj} {aux_dj} {poss_dj} {obj_dj} ye."
                pairs.append({
                    "instruction": f"Traduis en Dioula : {fr_p}",
                    "output": dj_p,
                    "source": "possession",
                })

    print(f"   → {len(pairs)} paires de possession générées")
    return pairs


def generer_questions() -> list:
    """Génère des questions avec les mots interrogatifs (G010)."""
    pairs = []

    questions = [
        # (fr, dioula)
        # Questions oui/non avec 'wa'
        ("Est-ce que tu vas bien ?",            "I ka kɛnɛ wa ?"),
        ("Est-ce qu'il est là ?",               "A bɛ yan wa ?"),
        ("Est-ce que tu manges ?",              "I bɛ dumu wa ?"),
        ("Est-ce que vous partez ?",            "Aw bɛ taa wa ?"),
        ("Est-ce qu'il a mangé ?",              "A ye dumu wa ?"),
        ("Est-ce que tu as de l'argent ?",      "Wari bɛ i fɛ wa ?"),
        ("Est-ce que le marché est ouvert ?",   "Sugu bɛ vu wa ?"),
        ("Est-ce qu'elle viendra ?",            "A bɛna na wa ?"),

        # Questions avec 'mun' (quoi/que)
        ("Que fais-tu ?",                       "I bɛ mun kɛ ?"),
        ("Qu'est-ce qu'il cherche ?",           "A bɛ mun ɲini ?"),
        ("Qu'est-ce qu'elle a acheté ?",        "A ye mun san ?"),
        ("Que voulez-vous ?",                   "Aw bɛ mun ɲini ?"),

        # Questions avec 'jɔn' (qui)
        ("Qui est là ?",                        "Jɔn bɛ yan ?"),
        ("Qui a parlé ?",                       "Jɔn ye kuma ?"),
        ("Qui vient ?",                         "Jɔn bɛ na ?"),

        # Questions avec 'di' (quel est)
        ("Quel est ton nom ?",                  "I tɔgɔ di ?"),
        ("Quel est ton travail ?",              "I ka baara di ?"),
        ("Quel est le prix ?",                  "Joli ye a joli ye ?"),

        # Questions avec 'yɔrɔ di' (où)
        ("Où est le marché ?",                  "Sugu bɛ yɔrɔ di ?"),
        ("Où vas-tu ?",                         "I bɛ taa yɔrɔ di ?"),
        ("Où habites-tu ?",                     "I bɛ yɔrɔ di sigi ?"),
        ("Où est ta maison ?",                  "I ka so bɛ yɔrɔ di ?"),

        # Questions avec 'waati di' (quand)
        ("Quand vas-tu venir ?",                "I bɛna na waati di ?"),
        ("Quand est-ce qu'il part ?",           "A bɛna taa waati di ?"),

        # Questions avec 'cogodɔ' (comment)
        ("Comment vas-tu ?",                    "I ka cogodɔ ?"),
        ("Comment fait-on ça ?",               "Mun cogodɔ an bɛ o kɛ ?"),
        ("Comment dit-on 'merci' en Dioula ?", "Cogodɔ an bɛ 'merci' fɔ julakan na ?"),
    ]

    for fr, dj in questions:
        pairs.append({
            "instruction": f"Traduis en Dioula : {fr}",
            "output": dj,
            "source": "questions",
        })
        pairs.append({
            "instruction": f"Traduis en français : {dj}",
            "output": fr,
            "source": "questions",
        })

    print(f"   → {len(pairs)} paires de questions générées")
    return pairs


def generer_copule() -> list:
    """Génère des phrases avec la copule 'ye' (G009) : X ye Y ye."""
    pairs = []

    identifications = [
        # (fr, dioula)
        ("Je suis étudiant.",               "Ne ye kalandenw ye."),
        ("Tu es commerçant.",               "I ye jula ye."),
        ("Il est médecin.",                 "A ye dɔgɔtɔrɔ ye."),
        ("Elle est enseignante.",           "A ye karamɔgɔ ye."),
        ("Nous sommes des amis.",           "An ye jɔw ye."),
        ("Abidjan est une grande ville.",   "Abidjan ye dugu ba ye."),
        ("Le dioula est une belle langue.", "Julakan ye kan duman ye."),
        ("Le riz est notre nourriture.",    "Malo ye an ka dɔni ye."),
        # Négatif avec tɛ
        ("Il n'est pas commerçant.",        "A tɛ jula ye."),
        ("Je ne suis pas médecin.",         "Ne tɛ dɔgɔtɔrɔ ye."),
        ("Ce n'est pas de l'eau.",          "O tɛ ji ye."),
        ("Elle n'est pas étudiante.",       "A tɛ kalandenw ye."),
    ]

    for fr, dj in identifications:
        pairs.append({
            "instruction": f"Traduis en Dioula : {fr}",
            "output": dj,
            "source": "copule",
        })
        pairs.append({
            "instruction": f"Traduis en français : {dj}",
            "output": fr,
            "source": "copule",
        })

    # Questions sur la règle
    pairs.append({
        "instruction": "Comment dit-on 'X est Y' en Dioula ?",
        "output": "En Dioula, on utilise la structure 'X ye Y ye'. Par exemple : 'Ne ye kalandenw ye' = 'Je suis étudiant'. Pour la négation, on dit 'X tɛ Y ye'.",
        "source": "copule",
    })

    print(f"   → {len(pairs)} paires de copule 'ye' générées")
    return pairs


def generer_explications_grammaticales() -> list:
    """Génère des paires Q&A sur les règles de grammaire pour que le modèle puisse les expliquer."""
    pairs = []

    qr = [
        (
            "Quelle est la structure de base d'une phrase en Dioula ?",
            "Le Dioula suit la structure SOV : Sujet + Auxiliaire + Objet + Verbe. C'est l'inverse du français (SVO). Par exemple : 'Je mange du riz' = 'Ne bɛ malo dumu' (Je [bɛ] riz manger)."
        ),
        (
            "C'est quoi l'auxiliaire 'bɛ' en Dioula ?",
            "'Bɛ' est l'auxiliaire du présent affirmatif en Dioula. Il se place entre le sujet et l'objet. Exemple : 'Ne bɛ malo dumu' = 'Je mange du riz'. Son contraire (négation) est 'tɛ'."
        ),
        (
            "Comment faire la négation au présent en Dioula ?",
            "Pour la négation au présent, on remplace 'bɛ' par 'tɛ'. Exemple : 'Ne bɛ dumu' (Je mange) → 'Ne tɛ dumu' (Je ne mange pas). La structure SOV reste la même."
        ),
        (
            "Comment conjuguer au passé en Dioula ?",
            "Au passé affirmatif, on utilise 'ye' à la place de 'bɛ'. Exemple : 'Ne ye malo dumu' = 'J'ai mangé du riz'. Pour le passé négatif, on utilise 'ma' : 'Ne ma dumu' = 'Je n'ai pas mangé'."
        ),
        (
            "Comment exprimer le futur en Dioula ?",
            "Le futur affirmatif utilise 'bɛna' : 'Ne bɛna taa' = 'Je vais partir'. Le futur négatif utilise 'tɛna' : 'Ne tɛna na' = 'Je ne viendrai pas'."
        ),
        (
            "Comment exprimer la possession en Dioula ?",
            "La possession se marque avec 'ka' entre le possesseur et l'objet : Possesseur + ka + Objet. Exemples : 'ne ka so' = 'ma maison', 'i ka fa' = 'ton père', 'a ka ba' = 'sa mère'."
        ),
        (
            "Est-ce qu'il y a une différence entre 'il' et 'elle' en Dioula ?",
            "Non ! En Dioula, 'a' désigne à la fois 'il' et 'elle'. Il n'y a pas de genre grammatical. Donc 'A bɛ baara kɛ' peut signifier 'Il travaille' ou 'Elle travaille' selon le contexte."
        ),
        (
            "Comment former le pluriel en Dioula ?",
            "Le pluriel se forme en ajoutant le suffixe '-w' au nom. Exemples : 'mɔgɔ' (personne) → 'mɔgɔw' (personnes), 'so' (maison) → 'sow' (maisons), 'den' (enfant) → 'denw' (enfants)."
        ),
        (
            "Comment poser une question oui/non en Dioula ?",
            "Pour une question oui/non, on ajoute 'wa' en fin de phrase. Exemple : 'I ka kɛnɛ wa ?' = 'Est-ce que tu vas bien ?' ou 'I bɛ dumu wa ?' = 'Est-ce que tu manges ?'."
        ),
        (
            "Quels sont les pronoms personnels en Dioula ?",
            "Les pronoms Dioula : 'ne/n' (je/me), 'i' (tu/te), 'a' (il/elle), 'an' (nous), 'aw' (vous), 'u' (ils/elles). Note : pas de distinction masculin/féminin, 'a' = il et elle."
        ),
        (
            "Comment dit-on 'être' en Dioula pour identifier quelque chose ?",
            "Pour identifier (X est Y), on utilise la copule 'ye' : structure 'X ye Y ye'. Exemple : 'Ne ye jula ye' = 'Je suis commerçant'. Pour la négation : 'Ne tɛ jula ye' = 'Je ne suis pas commerçant'."
        ),
        (
            "Quelle est la différence entre 'bɛ', 'ye' et 'bɛna' en Dioula ?",
            "Ce sont les trois auxiliaires temporels principaux : 'bɛ' marque le présent affirmatif, 'ye' marque le passé affirmatif, 'bɛna' marque le futur affirmatif. Leurs formes négatives sont respectivement 'tɛ', 'ma', 'tɛna'."
        ),
        (
            "Comment fonctionne l'ordre des mots en Dioula par rapport au français ?",
            "Le Dioula est SOV (Sujet-Objet-Verbe), le français est SVO (Sujet-Verbe-Objet). Exemple : 'Je mange du riz' (SVO) → 'Ne bɛ malo dumu' = littéralement 'Je [présent] riz manger' (SOV). L'objet vient AVANT le verbe en Dioula."
        ),
        (
            "Si quelqu'un me parle en Dioula comment je réponds ?",
            "Tu peux répondre en Dioula ! Voici les bases : 'Aw ni ce' (bonjour), 'N ka kɛnɛ' (je vais bien), 'I ni baaraka' (merci), 'Hali kana' (de rien), 'Ne tɛ a faamu' (je ne comprends pas), 'A fɔ kokura' (répète s'il te plaît)."
        ),
    ]

    for q, r in qr:
        pairs.append({
            "instruction": q,
            "output": r,
            "source": "explication_grammaire",
        })

    print(f"   → {len(pairs)} paires Q&A grammaticales générées")
    return pairs


def generer_depuis_json_source() -> list:
    """Charge et intègre les paires du dataset original."""
    pairs = []
    import os

    for base in [".", "/mnt/user-data/uploads"]:
        alpaca_path = os.path.join(base, "dioula_alpaca_v3_finetune.json")
        if os.path.exists(alpaca_path):
            with open(alpaca_path, encoding="utf-8") as f:
                alpaca = json.load(f)
            for e in alpaca:
                pairs.append({
                    "instruction": e["instruction"],
                    "output":      e["output"],
                    "source":      "alpaca_original",
                })
            print(f"   → {len(pairs)} paires du dataset Alpaca original")
            break

    return pairs


# ══════════════════════════════════════════════════════════════════
# FORMATAGE ET PIPELINE
# ══════════════════════════════════════════════════════════════════

def format_chat(instruction: str, output: str) -> dict:
    """Format messages compatible mlx-lm (toutes versions)."""
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": instruction},
            {"role": "assistant", "content": output},
        ]
    }


def main():
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n🔵 GÉNÉRATION DES DONNÉES GRAMMATICALES")
    print("=" * 55)

    all_pairs = []

    # 1. Dataset original Alpaca
    print("\n📂 Source 1 — Dataset Alpaca original")
    all_pairs += generer_depuis_json_source()

    # 2. Phrases SOV combinatoires
    print("\n📐 Source 2 — Phrases SOV (structure grammaticale)")
    all_pairs += generer_phrases_sov()

    # 3. Possession
    print("\n🏠 Source 3 — Possession ('ka')")
    all_pairs += generer_possession()

    # 4. Questions
    print("\n❓ Source 4 — Questions (wa, mun, jɔn, yɔrɔ di...)")
    all_pairs += generer_questions()

    # 5. Copule 'ye'
    print("\n🔗 Source 5 — Copule 'ye' (être / identification)")
    all_pairs += generer_copule()

    # 6. Explications grammaticales Q&A
    print("\n📚 Source 6 — Q&A explications grammaticales")
    all_pairs += generer_explications_grammaticales()

    print(f"\n   📊 Total brut : {len(all_pairs)} paires")

    # ── Dédoublonnage ──
    print("\n🔵 DÉDOUBLONNAGE")
    seen = set()
    unique = []
    dupes  = 0
    for p in all_pairs:
        key = (p["instruction"].strip().lower(), p["output"].strip().lower())
        if key not in seen:
            seen.add(key)
            unique.append(p)
        else:
            dupes += 1
    print(f"   → {dupes} doublons supprimés → {len(unique)} paires uniques")

    # ── Formatage ──
    print("\n🔵 FORMATAGE (messages format — mlx-lm compatible)")
    formatted = [format_chat(p["instruction"], p["output"]) for p in unique]

    # ── Split ──
    random.shuffle(formatted)
    n       = len(formatted)
    n_train = int(n * 0.80)
    n_valid = int(n * 0.10)

    splits = {
        "train": formatted[:n_train],
        "valid": formatted[n_train:n_train + n_valid],
        "test":  formatted[n_train + n_valid:],
    }

    # ── Écriture ──
    print("\n🔵 ÉCRITURE DES FICHIERS")
    for name, data in splits.items():
        path = OUTPUT_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"   ✅ {path} ({len(data)} exemples)")

    # ── Stats ──
    from collections import Counter
    sources = Counter(p.get("source", "?") for p in unique)

    print("\n" + "═" * 55)
    print("✅ DATASET PRÊT !")
    print("═" * 55)
    print(f"\n   Total : {len(unique)} paires uniques")
    print(f"   Train : {len(splits['train'])} | Valid : {len(splits['valid'])} | Test : {len(splits['test'])}")
    print("\n   Par source :")
    for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"     {src:<30} : {cnt}")

    print("\n🚀 Lance maintenant :")
    print()
    print('   mlx_lm.lora \\')
    print('       --model ./llama-3.2-3b-mlx \\')
    print('       --train \\')
    print('       --data "./data" \\')
    print('       --num-layers 8 \\')
    print('       --iters 1000 \\')
    print('       --batch-size 2 \\')
    print('       --learning-rate 1e-4 \\')
    print('       --val-batches 10')


if __name__ == "__main__":
    main()
