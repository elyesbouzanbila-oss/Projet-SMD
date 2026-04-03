"""
=============================================================================
  Évaluation du système de tatouage QIM — Benchmark Phase 2
  Reproduit les expériences de Lefevre et al. (XLIM)
=============================================================================
  Usage :
      python evaluate.py [--image path/to/image.png] [--bits 256]
=============================================================================
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse, os, time

from watermark import (
    encode, decode, attack_jpeg, attack_gaussian_noise,
    compute_ber, compute_psnr, compute_dwr, STEP, SEED
)

# ─────────────────────────────────────────────────────────────────────────────
#  UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def load_or_create_test_image(path=None, size=256):
    """Charge l'image fournie ou génère une image de test colorée."""
    if path and os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            print(f"[INFO] Image chargée : {path} — {img.shape}")
            return img
    # Image de test synthétique si aucune image n'est fournie
    print("[INFO] Génération d'une image de test synthétique (dégradé couleur)...")
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            img[i, j, 0] = int(255 * i / size)               # Canal B
            img[i, j, 1] = int(255 * j / size)               # Canal G
            img[i, j, 2] = int(255 * (1 - (i+j)/(2*size)))  # Canal R
    return img


def generate_watermark(n_bits, seed=SEED):
    """Génère un watermark binaire pseudo-aléatoire reproductible."""
    rng = np.random.default_rng(seed + 100)
    return rng.integers(0, 2, size=n_bits, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  EXPÉRIENCE 1 — Qualité visuelle (PSNR / DWR)
# ─────────────────────────────────────────────────────────────────────────────

def experiment_quality(image_bgr, n_bits):
    """
    Compare la qualité visuelle (PSNR, DWR) entre approche constante (GA)
    et approche adaptative (AA) sur l'image tatouée sans attaque.
    """
    print("\n" + "="*60)
    print("  EXPÉRIENCE 1 — Qualité visuelle (sans attaque)")
    print("="*60)

    wm_bits = generate_watermark(n_bits)
    results = {}

    for label, adaptive in [("GA (constante)", False), ("AA (adaptative)", True)]:
        t0 = time.time()
        wm_img, n_embedded = encode(image_bgr, wm_bits, adaptive=adaptive)
        elapsed = time.time() - t0

        psnr_val = compute_psnr(image_bgr, wm_img)
        dwr_val  = compute_dwr(image_bgr, wm_img)

        # Extraction sans attaque (BER doit être ≈ 0)
        extracted = decode(wm_img, n_embedded, adaptive=adaptive)
        ber_clean = compute_ber(wm_bits[:n_embedded], extracted)

        print(f"\n  [{label}]")
        print(f"    Bits insérés : {n_embedded}")
        print(f"    PSNR         : {psnr_val:.2f} dB")
        print(f"    DWR          : {dwr_val:.2f} dB")
        print(f"    BER (propre) : {ber_clean:.4f}")
        print(f"    Temps        : {elapsed:.2f}s")

        results[label] = {
            'image': wm_img, 'psnr': psnr_val, 'dwr': dwr_val,
            'ber_clean': ber_clean, 'n_bits': n_embedded
        }

    return results, wm_bits


# ─────────────────────────────────────────────────────────────────────────────
#  EXPÉRIENCE 2 — Robustesse sous compression JPEG
# ─────────────────────────────────────────────────────────────────────────────

def experiment_jpeg(image_bgr, wm_bits, n_bits,
                    quality_range=None):
    """
    Mesure le BER après compression JPEG pour plusieurs facteurs de qualité.
    Reproduit la Figure 6 de Lefevre et al.
    """
    if quality_range is None:
        quality_range = [70, 75, 80, 85, 90, 95, 100]

    print("\n" + "="*60)
    print("  EXPÉRIENCE 2 — Robustesse : compression JPEG")
    print("="*60)

    results = {'quality': quality_range, 'GA': [], 'AA': []}

    for label, adaptive, key in [("GA (constante)", False, 'GA'),
                                   ("AA (adaptative)", True, 'AA')]:
        wm_img, n_embedded = encode(image_bgr, wm_bits, adaptive=adaptive)
        bits_ref = wm_bits[:n_embedded]

        print(f"\n  [{label}] — {n_embedded} bits insérés")
        for q in quality_range:
            attacked = attack_jpeg(wm_img, q)
            extracted = decode(attacked, n_embedded, adaptive=adaptive)
            ber = compute_ber(bits_ref, extracted)
            results[key].append(ber)
            print(f"    q={q:3d} → BER = {ber:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  EXPÉRIENCE 3 — Robustesse sous bruit gaussien
# ─────────────────────────────────────────────────────────────────────────────

def experiment_gaussian(image_bgr, wm_bits, n_bits,
                        sigma_range=None):
    """
    Mesure le BER après ajout de bruit gaussien.
    Reproduit la Figure 7 de Lefevre et al.
    """
    if sigma_range is None:
        sigma_range = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    print("\n" + "="*60)
    print("  EXPÉRIENCE 3 — Robustesse : bruit gaussien")
    print("="*60)

    results = {'sigma': sigma_range, 'GA': [], 'AA': []}

    for label, adaptive, key in [("GA (constante)", False, 'GA'),
                                   ("AA (adaptative)", True, 'AA')]:
        wm_img, n_embedded = encode(image_bgr, wm_bits, adaptive=adaptive)
        bits_ref = wm_bits[:n_embedded]

        print(f"\n  [{label}] — {n_embedded} bits insérés")
        for sigma in sigma_range:
            attacked = attack_gaussian_noise(wm_img, sigma)
            extracted = decode(attacked, n_embedded, adaptive=adaptive)
            ber = compute_ber(bits_ref, extracted)
            results[key].append(ber)
            print(f"    σ={sigma:3d} → BER = {ber:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def save_visual_comparison(original, quality_results, output_dir):
    """
    Enregistre la comparaison visuelle GA vs AA
    (analogue aux Figures 3 & 4 de Lefevre et al.)
    """
    ga_img = quality_results["GA (constante)"]['image']
    aa_img = quality_results["AA (adaptative)"]['image']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Comparaison visuelle : Original | GA (constante) | AA (adaptative)",
                 fontsize=13, fontweight='bold')

    for ax, img, title in zip(
        axes,
        [original, ga_img, aa_img],
        ["Image originale",
         f"GA — PSNR={quality_results['GA (constante)']['psnr']:.1f} dB",
         f"AA — PSNR={quality_results['AA (adaptative)']['psnr']:.1f} dB"]
    ):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(output_dir, "fig1_comparaison_visuelle.png")
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[FIGURE] Sauvegardée → {path}")


def save_ber_curves(jpeg_res, gauss_res, output_dir):
    """
    Génère les courbes BER (reproduction des Figures 6 & 7 de Lefevre et al.)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Robustesse du tatouage QIM — Courbes BER\n"
                 "(Reproduction Figures 6 & 7 — Lefevre et al., XLIM)",
                 fontsize=13, fontweight='bold')

    # ── Figure 6 : JPEG ──
    ax1.plot(jpeg_res['quality'], jpeg_res['GA'], 'b-o', linewidth=2,
             markersize=6, label='STDM-QIM GA (constante)')
    ax1.plot(jpeg_res['quality'], jpeg_res['AA'], 'r-s', linewidth=2,
             markersize=6, label='STDM-QIM AA (adaptative)')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='BER=0.5 (aléatoire)')
    ax1.set_xlabel("Facteur de compression JPEG q", fontsize=11)
    ax1.set_ylabel("Bit Error Rate (BER)", fontsize=11)
    ax1.set_title("Attaque JPEG", fontsize=12)
    ax1.set_ylim(-0.02, 0.65)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Figure 7 : Bruit gaussien ──
    ax2.plot(gauss_res['sigma'], gauss_res['GA'], 'b-o', linewidth=2,
             markersize=6, label='STDM-QIM GA (constante)')
    ax2.plot(gauss_res['sigma'], gauss_res['AA'], 'r-s', linewidth=2,
             markersize=6, label='STDM-QIM AA (adaptative)')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='BER=0.5 (aléatoire)')
    ax2.set_xlabel("Écart-type du bruit gaussien σ", fontsize=11)
    ax2.set_ylabel("Bit Error Rate (BER)", fontsize=11)
    ax2.set_title("Attaque bruit gaussien", fontsize=12)
    ax2.set_ylim(-0.02, 0.75)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig2_courbes_BER.png")
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[FIGURE] Sauvegardée → {path}")


def save_difference_map(original, quality_results, output_dir):
    """Carte de différence amplifiée entre image originale et tatouée."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cartes de différence amplifiées (×10)\n"
                 "GA (constante) vs AA (adaptative)", fontsize=12, fontweight='bold')

    for ax, key, title in zip(
        axes,
        ["GA (constante)", "AA (adaptative)"],
        ["GA — différence ×10", "AA — différence ×10"]
    ):
        wm = quality_results[key]['image'].astype(float)
        orig = original.astype(float)
        diff = np.abs(wm - orig) * 10
        diff_rgb = cv2.cvtColor(np.clip(diff, 0, 255).astype(np.uint8),
                                cv2.COLOR_BGR2RGB)
        ax.imshow(diff_rgb)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(output_dir, "fig3_difference_maps.png")
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[FIGURE] Sauvegardée → {path}")


def save_summary_table(quality_results, jpeg_res, gauss_res, output_dir):
    """Tableau de synthèse des métriques clés."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis('off')

    ga = quality_results["GA (constante)"]
    aa = quality_results["AA (adaptative)"]

    # BER sous JPEG q=70 et Gauss σ=10
    ber_ga_jpeg70  = jpeg_res['GA'][0]   if jpeg_res['quality'][0] == 70  else jpeg_res['GA'][0]
    ber_aa_jpeg70  = jpeg_res['AA'][0]   if jpeg_res['quality'][0] == 70  else jpeg_res['AA'][0]
    sigma_idx = gauss_res['sigma'].index(10) if 10 in gauss_res['sigma'] else min(5, len(gauss_res['sigma'])-1)
    ber_ga_gauss10 = gauss_res['GA'][sigma_idx]
    ber_aa_gauss10 = gauss_res['AA'][sigma_idx]

    table_data = [
        ["Métrique", "GA (constante)", "AA (adaptative)", "Meilleur"],
        ["PSNR (dB)", f"{ga['psnr']:.2f}", f"{aa['psnr']:.2f}",
         "AA ✓" if aa['psnr'] >= ga['psnr'] else "GA ✓"],
        ["DWR (dB)", f"{ga['dwr']:.2f}", f"{aa['dwr']:.2f}", "—"],
        ["BER sans attaque", f"{ga['ber_clean']:.4f}", f"{aa['ber_clean']:.4f}",
         "AA ✓" if aa['ber_clean'] <= ga['ber_clean'] else "GA ✓"],
        [f"BER JPEG q={jpeg_res['quality'][0]}", f"{ber_ga_jpeg70:.4f}", f"{ber_aa_jpeg70:.4f}",
         "AA ✓" if ber_aa_jpeg70 <= ber_ga_jpeg70 else "GA ✓"],
        [f"BER Gauss σ=10", f"{ber_ga_gauss10:.4f}", f"{ber_aa_gauss10:.4f}",
         "AA ✓" if ber_aa_gauss10 <= ber_ga_gauss10 else "GA ✓"],
    ]

    colors_header = [["#1F4E79"] * 4]
    colors_rows   = [["#EBF3FB", "#EBF3FB", "#EBF3FB", "#EBF3FB"] if i % 2 == 0
                     else ["white"] * 4 for i in range(len(table_data) - 1)]

    tbl = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc='center',
        loc='center',
        cellColours=colors_rows,
        colColours=["#1F4E79"] * 4
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(color='white', fontweight='bold')
        if col == 3 and row > 0:
            cell.set_facecolor("#D4EDDA")

    ax.set_title("Tableau de synthèse — Benchmark QIM (Lefevre et al.)",
                 fontsize=13, fontweight='bold', pad=20, color="#1F4E79")

    plt.tight_layout()
    path = os.path.join(output_dir, "fig4_tableau_synthese.png")
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[FIGURE] Sauvegardée → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  POINT D'ENTRÉE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark du système de tatouage QIM — Mini-Projet L2-IRS")
    parser.add_argument('--image', type=str, default=None,
                        help='Chemin vers l\'image hôte (PNG/JPG)')
    parser.add_argument('--bits', type=int, default=128,
                        help='Nombre de bits du watermark (défaut: 128)')
    parser.add_argument('--step', type=float, default=STEP,
                        help=f'Pas de quantification Δ (défaut: {STEP})')
    parser.add_argument('--output', type=str, default='results',
                        help='Dossier de sortie (défaut: results/)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("="*60)
    print("  SYSTÈME DE TATOUAGE NUMÉRIQUE QIM — BENCHMARK")
    print("  Mini-Projet L2-IRS | Basé sur Lefevre et al. (XLIM)")
    print("="*60)
    print(f"  Pas Δ = {args.step} | Bits = {args.bits} | Seed = {SEED}")

    # Chargement image
    image = load_or_create_test_image(args.image)

    # Génération du watermark
    wm_bits = generate_watermark(args.bits)

    # Expériences
    quality_res, wm_bits = experiment_quality(image, args.bits)
    n_embedded = quality_res["GA (constante)"]['n_bits']

    jpeg_res  = experiment_jpeg(image, wm_bits, n_embedded)
    gauss_res = experiment_gaussian(image, wm_bits, n_embedded)

    # Visualisations
    print("\n" + "="*60)
    print("  GÉNÉRATION DES FIGURES")
    print("="*60)
    save_visual_comparison(image, quality_res, args.output)
    save_ber_curves(jpeg_res, gauss_res, args.output)
    save_difference_map(image, quality_res, args.output)
    save_summary_table(quality_res, jpeg_res, gauss_res, args.output)

    # Sauvegarde des images tatouées
    cv2.imwrite(os.path.join(args.output, "watermarked_GA.png"),
                quality_res["GA (constante)"]['image'])
    cv2.imwrite(os.path.join(args.output, "watermarked_AA.png"),
                quality_res["AA (adaptative)"]['image'])
    print(f"\n[INFO] Images tatouées sauvegardées dans {args.output}/")

    print("\n" + "="*60)
    print("  BENCHMARK TERMINÉ")
    print(f"  Résultats dans : {args.output}/")
    print("="*60)


if __name__ == "__main__":
    main()
