from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from plant_model import predict_plant
from disease_model import predict_disease

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
class PlantInput(BaseModel):
    lightLevel: str
    humidity: float
    temperature: float
    space: str
    wateringFrequency: float
    careLevel: str
    petSafe: str
    soilType: str
    fertilizerNeed: str

@app.post("/predict-plant")
async def plant_prediction(data: PlantInput):
    result = predict_plant(data.dict())
    return {"prediction": result}


@app.post("/predict-disease")
async def disease_prediction(file: UploadFile = File(...)):
    image = await file.read()
    disease, confidence = predict_disease(image)

    return {
        "disease": disease,
        "confidence": round(confidence*100,2)
    }