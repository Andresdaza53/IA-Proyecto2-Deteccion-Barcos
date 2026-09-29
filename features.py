"""
features.py — Preprocesamiento y descriptores visuales (compartido por train.py y app.py).

Usar EXACTAMENTE la misma función en entrenamiento e inferencia evita el
"training/serving skew": si la UI preprocesara distinto, el accuracy en vivo caería.

Descriptores:
  - HOG  : forma/bordes del casco (el barco es un objeto alargado con bordes fuertes).
  - COLOR: histograma HSV + media/desv. por canal RGB (agua oscura/uniforme vs. barco claro).
  - LBP  : textura local (oleaje, tierra, muelles vs. superficie lisa del barco).
"""
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern

IMG_SIZE = 80
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Parámetros de los descriptores (justificados en el informe, sección E2)
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                  block_norm="L2-Hys", feature_vector=True)
HSV_BINS = 16
LBP_P, LBP_R = 8, 1

ALL_BLOCKS = ("hog", "color", "lbp")


# ----------------------------------------------------------------------------- carga
def load_image(src) -> np.ndarray:
    """Lee ruta o archivo subido -> RGB float32 [0,1] de 80x80."""
    img = Image.open(src).convert("RGB")
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def label_from_name(name: str):
    """En ShipsNet el nombre empieza por '1__' (barco) o '0__' (no barco)."""
    n = Path(name).name
    if n.startswith("1__"):
        return 1
    if n.startswith("0__"):
        return 0
    return None


def list_images(folder) -> list:
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


# --------------------------------------------------------------------- aumentación
def augment(img: np.ndarray) -> list:
    """Vistas equivalentes: desde un satélite/drone un barco puede tener cualquier
    orientación, así que volteos y rotación de 90° son transformaciones válidas."""
    return [img, np.fliplr(img), np.flipud(img), np.rot90(img)]


# ---------------------------------------------------------------------- descriptores
def extract_blocks(img: np.ndarray) -> dict:
    """Devuelve cada descriptor por separado (permite el estudio de ablación)."""
    gray = rgb2gray(img)

    f_hog = hog(gray, **HOG_PARAMS)                                    # 2916

    hsv = rgb2hsv(img)
    hists = [np.histogram(hsv[..., c], bins=HSV_BINS, range=(0, 1), density=True)[0]
             / HSV_BINS for c in range(3)]
    stats = np.concatenate([img.reshape(-1, 3).mean(0), img.reshape(-1, 3).std(0)])
    f_color = np.concatenate(hists + [stats])                          # 54

    gray_u8 = (gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_u8, LBP_P, LBP_R, method="uniform")
    f_lbp, _ = np.histogram(lbp, bins=LBP_P + 2, range=(0, LBP_P + 2), density=True)  # 10

    return {"hog": f_hog.astype(np.float32),
            "color": f_color.astype(np.float32),
            "lbp": f_lbp.astype(np.float32)}


def extract_features(img: np.ndarray, blocks=ALL_BLOCKS) -> np.ndarray:
    b = extract_blocks(img)
    return np.concatenate([b[k] for k in blocks])
