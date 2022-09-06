import uvicorn
from fastapi import FastAPI,File, UploadFile
import torch
from training.model import GarbageClassifier
import cv2
import numpy as np
from PIL import Image

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    print(type(image))
    # image=cv2.imread(image)
    print(type(image))
    model=GarbageClassifier()
    model.load_state_dict(torch.load("./training/models/final-at-1.726.pt",map_location=torch.device('cpu')))
    print(model(image))
if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0")