from fastapi import FastAPI, UploadFile, File
import face_recognition
from PIL import Image
import numpy as np
from io import BytesIO
import uvicorn

app = FastAPI()

@app.post("/detect_faces")
async def detect_faces(file: UploadFile = File(...)):
    image_data = await file.read()
    image = face_recognition.load_image_file(BytesIO(image_data))
    locations = face_recognition.face_locations(image)

    cropped_faces = []
    for (top, right, bottom, left) in locations:
        margin = 30
        top = max(0, top - margin)
        left = max(0, left - margin)
        bottom = min(image.shape[0], bottom + margin)
        right = min(image.shape[1], right + margin)
        cropped = image[top:bottom, left:right]
        cropped_faces.append(cropped.tolist())  # convert numpy to list for JSON serialization

    return {"faces": cropped_faces, "count": len(cropped_faces)}
