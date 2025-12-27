from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from model import predict

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/detect")
async def predict_api(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    return predict(img)
