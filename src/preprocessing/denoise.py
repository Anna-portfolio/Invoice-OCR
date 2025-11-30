import cv2

#remove noise from a grayscale image
def remove_noise(gray_image, h=30):
    return cv2.fastNlMeansDenoising(gray_image, h=h)
