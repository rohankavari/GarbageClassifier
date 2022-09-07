import uvicorn
from fastapi import FastAPI, File, UploadFile
import torch
from training.model import GarbageClassifier
import cv2
import numpy as np
from PIL import Image
from training.predict import predict
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predictapi(file: UploadFile = File(...)):
    try:
        image = np.array(Image.open(file.file))
        print(type(image))
        result = await predict(image)
        print("Result", result.tolist())
        return {
            "message": "Success",
            "result": result.tolist()
        }
    except Exception as e:
        return {
            "message": "Error",
            "Error": str(e)
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
