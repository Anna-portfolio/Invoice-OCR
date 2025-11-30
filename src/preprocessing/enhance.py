import cv2
import numpy as np

def apply_gamma(img, gamma=1.0):
    # correct gamma 
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_clahe(img, clipLimit=2.0, tileGridSize=(8,8)):
    #apply CLAHE for contrast enchancement
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img)

def brighten_linear(img, brightness=40, contrast=1.05):
    #linear brightness adjustment using alpha and beta
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
