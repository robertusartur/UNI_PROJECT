from fastapi import FastAPI
from pydantic import BaseModel
from ml.model import load_staff

model = None
app = FastAPI()

class PredictParams(BaseModel):
    datetime: str
    geo_lat: float
    geo_lon: float
    building_type: int
    level: int
    levels: int
    rooms: int
    area: float
    kitchen_area: float
    object_type: int
    studio_apartment: int
    region: str

    class Config:
        schema_extra = {
            "example": {
                "datetime": "2021-08-01",
                "geo_lat": 59.805808,
                "geo_lon": 30.376141,
                "building_type": 1,
                "level": 1,
                "levels": 8,
                "rooms": 3,
                "area": 82.6,
                "kitchen_area": 10.8,
                "object_type": 1,
                "studio_apartment": 0,
                "region": "Санкт-Петербург"
            }
        }

@app.get("/")
def index():
    return {"params": "Cost"}

@app.on_event("startup")
def startup_event():
    global model
    model = load_staff()

@app.post("/predict")
def predict_cost(params: PredictParams):
    model_prediction = model(params.dict())
    
    response = model_prediction

    return response
