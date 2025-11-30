import cv2
from .denoise import remove_noise
from .enhance import apply_gamma, apply_clahe, brighten_linear
from .deskew import detect_skew_angle, deskew_image, normalize_orientation

def preprocess_invoice_pipeline(image_path, output_path=None, rotate_180=False):
    #load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"File not found: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #denoise
    denoised = remove_noise(gray)

    #correct gamma
    brightened = apply_gamma(denoised, gamma=1.0)

    #CLAHE and linear brightening adjustment
    enhanced = apply_clahe(brightened)
    enhanced = brighten_linear(enhanced, brightness=40, contrast=1.05)

    #deskew
    angle = detect_skew_angle(enhanced)
    deskewed = deskew_image(enhanced, angle)

    #normalize orientation
    final_img = normalize_orientation(deskewed, rotate_180=rotate_180)

    if output_path:
        cv2.imwrite(output_path, final_img)

    return final_img
