"""
=============================================================================
  Tatouage numérique par QIM — Mini-Projet L2-IRS
  Système complet : encodage, décodage, attaques, évaluation
=============================================================================
  Basé sur :
    - Chen & Wornell (2001) — QIM
    - Moulin & Koetter (2005) — STDM-QIM
    - Lefevre, Carré, Gaborit (XLIM) — Approche psychovisuelle couleur
=============================================================================
"""

import numpy as np
import cv2
from scipy.fft import dctn, idctn
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
#  PARAMÈTRES GLOBAUX
# ─────────────────────────────────────────────────────────────────────────────
BLOCK_SIZE  = 8          # Taille de bloc DCT
STEP        = 30.0       # Pas de quantification Δ  (contrôle DWR)
SEED        = 42         # Clé secrète pour sélection pseudo-aléatoire des coeffs
COEFF_RATIO = 0.15       # Fraction de coefficients de moyenne fréquence utilisés

# Constantes du modèle psychovisuel d'Alleysson (loi de Naka-Rushton)
ALPHA = np.array([1665.0, 1665.0, 226.0])   # Gains (αL, αM, αS)
X0    = np.array([66.0,   33.0,   0.16])    # États d'adaptation (L0, M0, S0)


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 1 — MODÈLE PSYCHOVISUEL (Alleysson / Lefevre et al.)
# ─────────────────────────────────────────────────────────────────────────────

def naka_rushton(X, X0_val):
    """Loi de Naka-Rushton : x = X / (X + X0)"""
    return X / (X + X0_val + 1e-8)


def rgb_to_lms(pixel_rgb):
    """
    Conversion approximative RGB -> LMS en utilisant la matrice de Hunt-Pointer-Estevez.
    Permet de travailler dans l'espace où le modèle d'Alleysson est défini.
    """
    M = np.array([
        [0.3811, 0.5783, 0.0402],
        [0.1967, 0.7244, 0.0782],
        [0.0241, 0.1288, 0.8444]
    ])
    return M @ pixel_rgb


def compute_optimal_direction(pixel_rgb):
    """
    Calcule le vecteur direction optimal uP pour un pixel couleur RGB.

    Principe (Lefevre et al. / Alleysson) :
    1. Convertir le pixel dans l'espace LMS
    2. Appliquer la transduction de Naka-Rushton pour obtenir les coordonnées lms
    3. Construire l'ellipsoïde d'iso-perception dans RGB via la jacobienne J
       de la transformation f : RGB -> lms
    4. Le vecteur propre associé à la valeur propre maximale de J^T J
       correspond à l'axe principal de l'ellipsoïde -> direction optimale uP

    Le vecteur direction optimal minimise la distorsion perçue par le SVH
    pour un bruit de quantification donné.
    """
    p = np.clip(pixel_rgb.astype(float), 1.0, 254.0)

    # Matrice Hunt-Pointer-Estevez (RGB -> LMS)
    M = np.array([
        [0.3811, 0.5783, 0.0402],
        [0.1967, 0.7244, 0.0782],
        [0.0241, 0.1288, 0.8444]
    ])
    lms_vals = M @ p  # valeurs LMS non transduites

    # Dérivée de Naka-Rushton : df/dX = X0 / (X + X0)^2
    deriv = X0 / (lms_vals + X0 + 1e-8) ** 2

    # Jacobienne J = diag(alpha * deriv) @ M
    # J_{ij} = alpha_i * d(f_i)/d(LMS_i) * M_{ij}
    J = np.diag(ALPHA * deriv) @ M  # shape (3, 3)

    # Valeurs singulières de J -> vecteur singulier à droite correspondant
    # à la valeur singulière max = axe principal de l'ellipsoïde
    _, _, Vt = np.linalg.svd(J)
    u_opt = Vt[0]  # premier vecteur singulier à droite

    # Normalisation
    norm = np.linalg.norm(u_opt)
    if norm < 1e-8:
        return np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    return u_opt / norm


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 2 — TRANSFORMATION DCT PAR BLOCS
# ─────────────────────────────────────────────────────────────────────────────

def select_mid_freq_coefficients(block_size, ratio, seed):
    """
    Sélectionne pseudo-aléatoirement des indices de coefficients DCT
    dans la bande de moyenne fréquence du bloc (hors DC et hautes fréquences).
    
    La moyenne fréquence offre le meilleur compromis invisibilité/robustesse
    face à la compression JPEG (les basses fréq. sont perceptibles,
    les hautes sont supprimées par JPEG).
    """
    rng = np.random.default_rng(seed)
    total = block_size * block_size

    # Zone de moyenne fréquence : indices 1/4 à 3/4 en ordre zigzag
    low_cut  = total // 4
    high_cut = total * 3 // 4

    # Parcours zigzag pour identifier les coefficients
    zigzag_indices = []
    for s in range(2 * block_size - 1):
        if s % 2 == 0:
            r_start = min(s, block_size - 1)
            c_start = max(0, s - block_size + 1)
            while r_start >= 0 and c_start < block_size:
                zigzag_indices.append((r_start, c_start))
                r_start -= 1
                c_start += 1
        else:
            c_start = min(s, block_size - 1)
            r_start = max(0, s - block_size + 1)
            while c_start >= 0 and r_start < block_size:
                zigzag_indices.append((r_start, c_start))
                r_start += 1
                c_start -= 1

    mid_freq = zigzag_indices[low_cut:high_cut]
    n_select = max(1, int(len(mid_freq) * ratio))
    chosen = rng.choice(len(mid_freq), size=n_select, replace=False)
    return [mid_freq[i] for i in sorted(chosen)]


def image_to_dct_blocks(channel_float, block_size=BLOCK_SIZE):
    """Découpe un canal en blocs et applique la DCT 2D sur chaque bloc."""
    h, w = channel_float.shape
    h_pad = (block_size - h % block_size) % block_size
    w_pad = (block_size - w % block_size) % block_size
    padded = np.pad(channel_float, ((0, h_pad), (0, w_pad)), mode='reflect')
    H, W = padded.shape
    blocks = padded.reshape(H // block_size, block_size,
                            W // block_size, block_size)
    blocks = blocks.transpose(0, 2, 1, 3)  # (nH, nW, bs, bs)
    dct_blocks = np.zeros_like(blocks)
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            dct_blocks[i, j] = dctn(blocks[i, j], norm='ortho')
    return dct_blocks, (h, w), (H, W)


def dct_blocks_to_image(dct_blocks, orig_shape, padded_shape, block_size=BLOCK_SIZE):
    """Reconstruit l'image depuis les blocs DCT."""
    H, W = padded_shape
    h, w = orig_shape
    result_blocks = np.zeros_like(dct_blocks)
    for i in range(dct_blocks.shape[0]):
        for j in range(dct_blocks.shape[1]):
            result_blocks[i, j] = idctn(dct_blocks[i, j], norm='ortho')
    reconstructed = result_blocks.transpose(0, 2, 1, 3)
    reconstructed = reconstructed.reshape(H, W)
    return reconstructed[:h, :w]


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 3 — ENCODEUR QIM (insertion de la marque)
# ─────────────────────────────────────────────────────────────────────────────

def qim_quantize(value, delta, bit):
    """
    Quantification QIM scalaire.
    Quantifie 'value' vers le treillis pair (bit=0) ou impair (bit=1).
    Q_m(x, Δ) = Δ * round(x/Δ - m/2) + m*Δ/2
    """
    q = delta * (np.round(value / delta - bit / 2.0) + bit / 2.0)
    return q


def encode(image_bgr, watermark_bits, step=STEP, seed=SEED,
           adaptive=True, coeff_ratio=COEFF_RATIO):
    """
    Insère un watermark binaire dans une image couleur via STDM-QIM.

    Args:
        image_bgr    : image hôte (uint8, BGR, H×W×3)
        watermark_bits : tableau de bits à insérer (0/1), longueur N_bits
        step         : pas de quantification Δ
        seed         : clé secrète
        adaptive     : True = approche adaptative AA (vecteur uP par pixel)
                       False = approche constante GA (u = (1,1,1)/√3)
        coeff_ratio  : fraction des coefficients de moyenne fréquence utilisés

    Returns:
        watermarked_bgr : image tatouée (uint8, BGR)
        n_bits_embedded : nombre de bits effectivement insérés
    """
    img_float = image_bgr.astype(np.float64)
    h, w = img_float.shape[:2]

    # Sélection des coefficients DCT (même seed = même sélection au décodage)
    coeff_positions = select_mid_freq_coefficients(BLOCK_SIZE, coeff_ratio, seed)
    n_coeffs_per_block = len(coeff_positions)

    # Calcul DCT par canal
    dct_channels = []
    orig_shapes  = []
    pad_shapes   = []
    for c in range(3):
        dct_b, orig_s, pad_s = image_to_dct_blocks(img_float[:, :, c])
        dct_channels.append(dct_b)
        orig_shapes.append(orig_s)
        pad_shapes.append(pad_s)

    # Nombre total de coefficients disponibles
    nH, nW = dct_channels[0].shape[:2]
    total_coeffs = nH * nW * n_coeffs_per_block
    n_bits = min(len(watermark_bits), total_coeffs)
    watermark_bits = watermark_bits[:n_bits]

    bit_idx = 0
    rng = np.random.default_rng(seed + 1)  # Pour la direction u (STDM)

    for bi in range(nH):
        for bj in range(nW):
            for (r, c) in coeff_positions:
                if bit_idx >= n_bits:
                    break

                # Pixel représentatif du bloc (centre) pour le vecteur adaptatif
                pi = min(int((bi + 0.5) * BLOCK_SIZE), h - 1)
                pj = min(int((bj + 0.5) * BLOCK_SIZE), w - 1)
                pixel_rgb = img_float[pi, pj, ::-1]  # BGR->RGB

                if adaptive:
                    u = compute_optimal_direction(pixel_rgb)
                else:
                    u = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)

                # Vecteur des 3 coefficients DCT correspondants (un par canal)
                coeff_vec = np.array([dct_channels[ch][bi, bj, r, c]
                                      for ch in range(3)])

                # STDM-QIM : projeter sur u, quantifier, reconstruire
                S = np.dot(coeff_vec, u)                          # projection scalaire
                bit = int(watermark_bits[bit_idx])
                S_q = qim_quantize(S, step, bit)                  # quantification QIM
                delta_coeff = (S_q - S) * u                       # modification vectorielle

                for ch in range(3):
                    dct_channels[ch][bi, bj, r, c] += delta_coeff[ch]

                bit_idx += 1
            if bit_idx >= n_bits:
                break
        if bit_idx >= n_bits:
            break

    # Reconstruction depuis DCT
    result = np.zeros_like(img_float)
    for c in range(3):
        result[:, :, c] = dct_blocks_to_image(
            dct_channels[c], orig_shapes[c], pad_shapes[c])

    watermarked = np.clip(result, 0, 255).astype(np.uint8)
    return watermarked, n_bits


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 4 — DÉCODEUR QIM (extraction de la marque)
# ─────────────────────────────────────────────────────────────────────────────

def qim_decode_bit(value, delta):
    """
    Décodage QIM : détermine le bit encodé en cherchant le treillis le plus proche.
    bit = argmin_m |value - Q_m(value, delta)|
    """
    q0 = qim_quantize(value, delta, 0)
    q1 = qim_quantize(value, delta, 1)
    return 0 if abs(value - q0) < abs(value - q1) else 1


def decode(image_bgr, n_bits, step=STEP, seed=SEED,
           adaptive=True, coeff_ratio=COEFF_RATIO):
    """
    Extrait le watermark d'une image tatouée (potentiellement attaquée).

    Returns:
        extracted_bits : tableau numpy des bits extraits (0/1)
    """
    img_float = image_bgr.astype(np.float64)
    h, w = img_float.shape[:2]

    coeff_positions = select_mid_freq_coefficients(BLOCK_SIZE, coeff_ratio, seed)

    dct_channels = []
    orig_shapes  = []
    pad_shapes   = []
    for c in range(3):
        dct_b, orig_s, pad_s = image_to_dct_blocks(img_float[:, :, c])
        dct_channels.append(dct_b)
        orig_shapes.append(orig_s)
        pad_shapes.append(pad_s)

    nH, nW = dct_channels[0].shape[:2]
    extracted = []
    bit_idx = 0

    for bi in range(nH):
        for bj in range(nW):
            for (r, c) in coeff_positions:
                if bit_idx >= n_bits:
                    break

                pi = min(int((bi + 0.5) * BLOCK_SIZE), h - 1)
                pj = min(int((bj + 0.5) * BLOCK_SIZE), w - 1)
                pixel_rgb = img_float[pi, pj, ::-1]

                if adaptive:
                    u = compute_optimal_direction(pixel_rgb)
                else:
                    u = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)

                coeff_vec = np.array([dct_channels[ch][bi, bj, r, c]
                                      for ch in range(3)])
                S = np.dot(coeff_vec, u)
                bit = qim_decode_bit(S, step)
                extracted.append(bit)
                bit_idx += 1

            if bit_idx >= n_bits:
                break
        if bit_idx >= n_bits:
            break

    return np.array(extracted[:n_bits], dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 5 — ATTAQUES
# ─────────────────────────────────────────────────────────────────────────────

def attack_jpeg(image_bgr, quality):
    """Compression JPEG avec facteur de qualité q ∈ [0, 100]."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', image_bgr, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def attack_gaussian_noise(image_bgr, sigma):
    """Ajout de bruit gaussien additif d'écart-type σ."""
    noise = np.random.normal(0, sigma, image_bgr.shape)
    noisy = image_bgr.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 6 — MÉTRIQUES
# ─────────────────────────────────────────────────────────────────────────────

def compute_ber(original_bits, extracted_bits):
    """BER = fraction de bits incorrectement décodés."""
    n = min(len(original_bits), len(extracted_bits))
    return np.mean(original_bits[:n] != extracted_bits[:n])


def compute_psnr(original_bgr, watermarked_bgr):
    """PSNR en dB entre image originale et image tatouée."""
    return psnr(original_bgr, watermarked_bgr, data_range=255)


def compute_dwr(original_bgr, watermarked_bgr):
    """
    DWR (Document-to-Watermark Ratio) en dB.
    DWR = 10 * log10(var(image) / var(bruit_de_tatouage))
    """
    diff = original_bgr.astype(float) - watermarked_bgr.astype(float)
    var_img  = np.var(original_bgr.astype(float))
    var_diff = np.var(diff)
    if var_diff < 1e-10:
        return float('inf')
    return 10 * np.log10(var_img / var_diff)
