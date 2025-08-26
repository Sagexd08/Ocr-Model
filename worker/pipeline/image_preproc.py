

from typing import Tuple
import cv2
import numpy as np
from PIL import Image


def to_cv(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def preprocess_image(image: Image.Image) -> Image.Image:
    img = to_cv(image)
    img = denoise_image(img)
    img = enhance_contrast(img)
    img = adaptive_binarize(img)
    img = remove_small_noise(img)
    img = deskew_image(img)
    return to_pil(img)


def deskew_image(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))
    if coords.size == 0:
        return img
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def denoise_image(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def adaptive_binarize(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 35, 15)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def remove_small_noise(img: np.ndarray, min_area: int = 30) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(255 - gray, connectivity=8)
    sizes = stats[1:, -1]
    img_clean = np.copy(gray)
    for i in range(0, nb_components - 1):
        if sizes[i] < min_area:
            img_clean[output == i + 1] = 255
    return cv2.cvtColor(img_clean, cv2.COLOR_GRAY2BGR)

