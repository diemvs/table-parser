from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
from ultralytics import YOLO

app = FastAPI()
model = YOLO("weights/yolo/weights.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    results = model(image)

    boxes = []
    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    return JSONResponse(content={"boxes": boxes})
