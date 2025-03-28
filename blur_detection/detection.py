import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\62086\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


def fix_image_size(image: np.array, expected_pixels: float = 2E6):
    """Resizes image while keeping aspect ratio."""
    ratio = np.sqrt(expected_pixels / (image.shape[0] * image.shape[1]))
    return cv2.resize(image, (0, 0), fx=ratio, fy=ratio)


def estimate_blur(image: np.array, threshold: int = 100):
    """Estimates the blur level using Laplacian variance."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = np.var(blur_map)  # Variance of Laplacian
    return blur_map, score, bool(score < threshold)


def pretty_blur_map(blur_map: np.array, sigma: int = 5, min_abs: float = 0.5):
    """Generates a visually appealing blur map."""
    abs_image = np.abs(blur_map).astype(np.float32)
    abs_image[abs_image < min_abs] = min_abs

    abs_image = np.log(abs_image)
    cv2.blur(abs_image, (sigma, sigma))
    return cv2.medianBlur(abs_image, sigma)


def extract_text(image: np.array):
    """Extracts text from the image using Tesseract OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    extracted_text = text_data['text']
    conf_scores = [conf for conf in text_data['conf'] if conf != -1]  # Ignore invalid confidence scores

    avg_confidence = sum(conf_scores) / len(conf_scores) if conf_scores else 0
    return " ".join(extracted_text).strip(), avg_confidence
