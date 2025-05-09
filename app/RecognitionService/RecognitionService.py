from ocr_enums import OCREngine, OCRLang
from fastapi import FastAPI, File, UploadFile, Query, Request
from fastapi.responses import JSONResponse
from typing import List, Tuple
from io import BytesIO
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
import easyocr
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(debug=True)
Instrumentator().instrument(app).expose(app)

def extract_cells(image: np.ndarray, min_cell_width: int = 20, min_cell_height: int = 20) -> List[List[Tuple[int, int, int, int]]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    vertical_lines = cv2.erode(binary, vertical_kernel, iterations=1)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    horizontal_lines = cv2.erode(binary, horizontal_kernel, iterations=1)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)

    table_structure = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    contours, _ = cv2.findContours(table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > min_cell_width and h > min_cell_height:
            cells.append((x, y, w, h))

    image_area = image.shape[0] * image.shape[1]
    cells = [cell for cell in cells if cell[2] * cell[3] < 0.5 * image_area]
    cells = sorted(cells, key=lambda b: (b[1], b[0]))

    rows = []
    current_row = []
    last_y = -100
    tolerance = 10
    for cell in cells:
        x, y, w, h = cell
        if abs(y - last_y) > tolerance:
            if current_row:
                rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [cell]
            last_y = y
        else:
            current_row.append(cell)
    if current_row:
        rows.append(sorted(current_row, key=lambda b: b[0]))

    return rows


def recognize_cells_with_tesseract(image: np.ndarray, rows: List[List[Tuple[int, int, int, int]]], lang: str = 'eng') -> pd.DataFrame:
    data = []
    for row in rows:
        row_data = []
        for (x, y, w, h) in row:
            margin = 2
            roi = image[y+margin:y+h-margin, x+margin:x+w-margin]
            roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, roi_bin = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(roi_bin, config='--psm 6', lang=lang).strip()
            row_data.append(text)
        data.append(row_data)
    return pd.DataFrame(data)


def recognize_cells_with_easyocr(image: np.ndarray, rows: List[List[Tuple[int, int, int, int]]], lang: str = 'ru') -> pd.DataFrame:
    reader = easyocr.Reader([lang], gpu=True)
    data = []
    for row in rows:
        row_data = []
        for (x, y, w, h) in row:
            margin = 2
            roi = image[y+margin:y+h-margin, x+margin:x+w-margin]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = reader.readtext(roi_rgb, detail=0)
            text = results[0] if results else ""
            row_data.append(text)
        data.append(row_data)
    return pd.DataFrame(data)

@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...), 
    lang: OCRLang = Query(default=OCRLang.eng), 
    engine: OCREngine = Query(default=OCREngine.tesseract)
):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        rows = extract_cells(image_np)

        if engine == OCREngine.easyocr:
            df = recognize_cells_with_easyocr(image_np, rows, lang=lang)
        elif engine == OCREngine.tesseract:
            df = recognize_cells_with_tesseract(image_np, rows, lang=lang)
        else:
            raise ValueError(f"Unsupported OCR engine: {engine}")

        table_data = df.fillna("").values.tolist()

        return {"table": table_data, "cell_count": sum(len(row) for row in rows)}
    except Exception as e:
        print('Recognition error:', e)
        return JSONResponse(status_code=500, content={"error": str(e)})
