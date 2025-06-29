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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import logging
from transformers import T5ForConditionalGeneration, GenerationConfig, T5Tokenizer
import torch
from torchvision import transforms
from prometheus_client import Summary
from time import perf_counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
rec_duration = Summary("recognition_duration_seconds", "Recognition processing duration in seconds")

tokenizer_for_restoring = T5Tokenizer.from_pretrained('bond005/ruT5-ASR-large')
model_for_restoring = T5ForConditionalGeneration.from_pretrained('bond005/ruT5-ASR-large')
config_for_restoring = GenerationConfig.from_pretrained('bond005/ruT5-ASR-large')

# Загрузка модели TrOCR
processor_trocr = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model_trocr = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")


trocr_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((384, 384)),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"device: {device}")
model_trocr.to(device)

if torch.cuda.is_available():
    model_for_restoring = model_for_restoring.cuda()

app = FastAPI(debug=True)
Instrumentator().instrument(app).expose(app)

def extract_cells(
    image: np.ndarray, 
    min_cell_width: int = 20, 
    min_cell_height: int = 20
) -> List[List[Tuple[int, int, int, int]]]:
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


def recognize_cells_with_tesseract(
    image: np.ndarray, 
    rows: List[List[Tuple[int, int, int, int]]], 
    lang: OCRLang = OCRLang.rus,
    correct_text: bool = True
) -> pd.DataFrame:
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
            
            if(correct_text):
                text = restore_text(text, tokenizer_for_restoring, config_for_restoring, model_for_restoring) if lang == OCRLang.rus else text
            
            row_data.append(text)
        data.append(row_data)
    return pd.DataFrame(data)

def recognize_cells_with_easyocr(
    image: np.ndarray, 
    rows: List[List[Tuple[int, int, int, int]]], 
    lang: str = OCRLang.rus,
    correct_text: bool = True
) -> pd.DataFrame:
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
            
            print(f"[OCR] Raw: {text}")
            
            if(correct_text):
                text = restore_text(text, tokenizer_for_restoring, config_for_restoring, model_for_restoring) if lang == OCRLang.rus else text
                print(f"[Corrected] {text}")
                
            row_data.append(text)
        data.append(row_data)
    return pd.DataFrame(data)

def restore_text(
    text: str, 
    tokenizer: T5Tokenizer, 
    config: GenerationConfig,
    model: T5ForConditionalGeneration
) -> str:
    if len(text) == 0:
        return ''
    x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
    max_size = int(x.input_ids.shape[1] * 2.0 + 10)
    min_size = 3
    if x.input_ids.shape[1] <= min_size:
        return text
    out = model.generate(**x, generation_config=config, max_length=max_size)
    res = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    return ' '.join(res.split())

@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...), 
    lang: OCRLang = Query(default=OCRLang.rus), 
    correct_text: bool = True,
    engine: OCREngine = Query(default=OCREngine.tesseract)
):
    start = perf_counter()
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        rows = extract_cells(image_np)

        if engine == OCREngine.easyocr:
            df = recognize_cells_with_easyocr(image_np, rows, lang=lang, correct_text=correct_text)
        elif engine == OCREngine.tesseract:
            df = recognize_cells_with_tesseract(image_np, rows, lang=lang, correct_text=correct_text)
        else:
            raise ValueError(f"Unsupported OCR engine: {engine}")

        table_data = df.fillna("").values.tolist()

        return {"table": table_data, "cell_count": sum(len(row) for row in rows)}
    except Exception as e:
        print('Recognition error:', e)
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        duration = perf_counter() - start
        rec_duration.observe(duration)

