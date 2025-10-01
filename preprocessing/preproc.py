# preprocessing/preproc.py
import cv2
import numpy as np
from PIL import Image
from .deskew import deskew

def preprocess_for_print_soft(pil_img: Image.Image) -> Image.Image:
    img = np.array(pil_img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.medianBlur(img, 3)
    img = deskew(img)
    return Image.fromarray(img)

def preprocess_for_print_hard(pil_img: Image.Image) -> Image.Image:
    img = np.array(pil_img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.medianBlur(img, 3)
    img = deskew(img)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return Image.fromarray(img)
