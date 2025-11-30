import cv2
import numpy as np

def detect_skew_angle(img):
    # Detect skew angle and return it
    binary = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 10
    )
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -45 < angle < 45:
                angles.append(angle)
    return np.median(angles) if angles else 0

def deskew_image(img, angle):
    # based on previously detected angle, rotate image to it
    h, w = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def normalize_orientation(img, rotate_180=False):
    #check if image is upright, 
    # rotate 90 degrees if width > height
    # optional 180 degrees
    h, w = img.shape
    if w > h:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotate_180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    return img
