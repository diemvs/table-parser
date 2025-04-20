from enum import Enum

class OCREngine(str, Enum):
    tesseract = "tesseract"
    easyocr = "easyocr"

class OCRLang(str, Enum):
    rus = "rus"
    eng = "eng"