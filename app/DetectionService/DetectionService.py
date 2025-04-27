from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
from ultralytics import YOLO
import logging
import torch
from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"device: {device}")

app = FastAPI()
model = YOLO("weights/yolo/weights.pt")
model.to(device)
Instrumentator().instrument(app).expose(app)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    results = model(image)

    boxes = []
    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    return JSONResponse(content={"boxes": boxes})
