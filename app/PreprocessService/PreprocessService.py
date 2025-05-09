from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
from PIL import Image
import base64
import io

app = FastAPI()

@app.post("/preprocess/pdf")
async def preprocess_pdf(file: UploadFile = File(...)):
    content = await file.read()

    images = convert_from_bytes(content, dpi=200)

    image_b64_list = []
    for img in images:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_b64_list.append(encoded)

    return JSONResponse(content={"images": image_b64_list})