import requests
import cv2
import numpy as np
from PIL import Image, ImageFilter
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt

from app.RecognitionService.ocr_enums.ocr_enums import OCREngine, OCRLang

def get_table_data(image_path: str, correct_text: bool = True, blur: bool = False, lang: OCRLang = OCRLang.rus):
    detect_url = "http://localhost:8000/detect"
    recognize_url = "http://localhost:8001/recognize"

    image = cv2.imread(image_path)

    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/jpeg")}
        detect_response = requests.post(detect_url, files=files)

    detect_response.raise_for_status()
    boxes = detect_response.json()["boxes"]

    results = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

        # Вырезаем ROI и кодируем в память
        roi = image[y1:y2, x1:x2]
        is_success, buffer = cv2.imencode(".jpg", roi)
        if not is_success:
            print(f"Ошибка при кодировании ROI {i}")
            continue

        roi_bytes = BytesIO(buffer.tobytes())

        files = {"file": ("cell.jpg", roi_bytes, "image/jpeg")}
        params = {"lang": lang, "engine": OCREngine.tesseract, "correct_text": correct_text}
        rec_response = requests.post(recognize_url, files=files, params=params)

        if rec_response.status_code == 200:
            data = rec_response.json()
            df = pd.DataFrame(data.get("table", []))
            results.append(df)
        else:
            print(f"[Ошибка OCR: {rec_response}]")
            print(rec_response.content)
            
    return results
    
def show_image(image_path: str):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()
    
def show_and_get_tables(image_path: str, correct_text: bool = True, blur: bool = False, lang: OCRLang = OCRLang.rus):
    try:
        show_image(image_path)
        results = get_table_data(image_path, correct_text, blur, lang)
        print(f"count of tables: {len(results)}")
        return results
            
    except Exception as e:
        print(e)
        
def save_data_frame(df: pd.DataFrame, file_name: str):
    df.to_csv(file_name, encoding="utf-8-sig", index=False, sep=";")