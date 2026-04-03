# Tatouage Numérique par QIM — Mini-Projet L2-IRS

Système complet de tatouage numérique d'images couleur basé sur **STDM-QIM**,
avec approche psychovisuelle adaptative (modèle d'Alleysson).

## Structure
```
watermark_qim/
├── watermark.py    ← Moteur principal (encodage, décodage, attaques, métriques)
├── evaluate.py     ← Benchmark complet (reproduit Lefevre et al.)
├── demo.py         ← Démo rapide en ligne de commande
└── README.md
```

## Installation
```bash
pip install opencv-python numpy matplotlib scikit-image scipy
```

## Utilisation rapide
```bash
# Démo rapide (image synthétique)
python demo.py

# Benchmark complet (image synthétique)
python evaluate.py

# Benchmark sur votre propre image
python evaluate.py --image votre_image.png --bits 256

# Options avancées
python evaluate.py --image image.png --bits 128 --step 25 --output mes_resultats
```

## Paramètres clés
| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `--bits`  | 128    | Nombre de bits du watermark |
| `--step`  | 30.0   | Pas de quantification Δ (↑ = + robuste, ↓ = + invisible) |
| `--output`| results| Dossier de sortie |

## Méthodes implémentées
- **GA** (Global Approach) : vecteur direction constant u=(1,1,1)/√3
- **AA** (Adaptive Approach) : vecteur direction uP optimisé par pixel via modèle Alleysson

## Métriques calculées
- **PSNR** (Peak Signal-to-Noise Ratio) — qualité visuelle
- **BER** (Bit Error Rate) — robustesse extraction
- **DWR** (Document-to-Watermark Ratio) — niveau de distorsion

## Références
- Chen & Wornell (2001) — QIM
- Moulin & Koetter (2005) — STDM-QIM
- Lefevre, Carré, Gaborit (XLIM) — Approche psychovisuelle couleur
- Alleysson & Méary (2012) — Modèle psychovisuel SVH
