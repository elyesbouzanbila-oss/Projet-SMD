"""
=============================================================================
  Démo rapide — Tatouage QIM (Mini-Projet L2-IRS)
=============================================================================
  Exemple d'utilisation du module watermark.py étape par étape.
  Lance : python demo.py
=============================================================================
"""

import numpy as np
import cv2
import os

from watermark import (
    encode, decode, attack_jpeg, attack_gaussian_noise,
    compute_ber, compute_psnr, compute_dwr
)


def demo():
    print("="*55)
    print("  DÉMO — Tatouage QIM | L2-IRS Mini-Projet")
    print("="*55)

    # ── 1. Créer une image de test (256×256, dégradé coloré) ──
    size = 256
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            img[i, j] = [int(255*i/size), int(255*j/size),
                         int(255*(1 - (i+j)/(2*size)))]

    print(f"\n[1] Image hôte : {img.shape}, dtype={img.dtype}")

    # ── 2. Watermark binaire ──
    np.random.seed(42)
    n_bits   = 64
    wm_bits  = np.random.randint(0, 2, size=n_bits, dtype=np.uint8)
    print(f"[2] Watermark  : {n_bits} bits | ex: {wm_bits[:8]}...")

    # ── 3. Encodage ──
    print("\n[3] ENCODAGE")
    for label, adaptive in [("GA (constante)", False), ("AA (adaptative)", True)]:
        wm_img, n_emb = encode(img, wm_bits, adaptive=adaptive)

        psnr_v = compute_psnr(img, wm_img)
        dwr_v  = compute_dwr(img, wm_img)

        # Décodage propre
        extracted = decode(wm_img, n_emb, adaptive=adaptive)
        ber_clean = compute_ber(wm_bits[:n_emb], extracted)

        print(f"\n  [{label}]")
        print(f"    PSNR         : {psnr_v:.2f} dB")
        print(f"    DWR          : {dwr_v:.2f} dB")
        print(f"    BER propre   : {ber_clean:.4f}  {'✓' if ber_clean == 0 else '!'}")

    # ── 4. Attaques ──
    print("\n[4] ATTAQUES (approche adaptative AA)")
    wm_aa, n_emb = encode(img, wm_bits, adaptive=True)
    bits_ref = wm_bits[:n_emb]

    print("\n  → Compression JPEG :")
    for q in [70, 80, 90, 100]:
        att = attack_jpeg(wm_aa, q)
        ext = decode(att, n_emb, adaptive=True)
        ber = compute_ber(bits_ref, ext)
        bar = "█" * int(ber * 20) + "░" * (20 - int(ber * 20))
        print(f"    q={q:3d}  BER={ber:.3f}  |{bar}|")

    print("\n  → Bruit gaussien :")
    for sigma in [0, 5, 10, 15, 20]:
        att = attack_gaussian_noise(wm_aa, sigma)
        ext = decode(att, n_emb, adaptive=True)
        ber = compute_ber(bits_ref, ext)
        bar = "█" * int(ber * 20) + "░" * (20 - int(ber * 20))
        print(f"    σ={sigma:3d}  BER={ber:.3f}  |{bar}|")

    # ── 5. Sauvegarde ──
    os.makedirs("results", exist_ok=True)
    cv2.imwrite("results/original.png", img)
    wm_ga, _ = encode(img, wm_bits, adaptive=False)
    wm_aa, _ = encode(img, wm_bits, adaptive=True)
    cv2.imwrite("results/watermarked_GA.png", wm_ga)
    cv2.imwrite("results/watermarked_AA.png", wm_aa)
    print("\n[5] Images sauvegardées dans results/")

    print("\n" + "="*55)
    print("  Pour le benchmark complet : python evaluate.py")
    print("="*55)


if __name__ == "__main__":
    demo()
