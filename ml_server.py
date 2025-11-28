from fastapi import FastAPI
import joblib
import numpy as np
from PIL import Image
app = FastAPI()

model = joblib.load("plant_health_model_xgb.pkl")
encoder = joblib.load("label_encoder.pkl")




@app.post("/predict")
def predict(Soil_Moisture: float, Ambient_Temperature: float, Soil_Temperature: float,Humidity: float, Light_Intensity: float, Soil_pH: float,Nitrogen_Level: float, Phosphorus_Level: float, Potassium_Level: float,Chlorophyll_Content:float):
    Electrochemical_Signal=-140
    X = np.array([[Soil_Moisture,Ambient_Temperature,Soil_Temperature,Humidity,Light_Intensity,Soil_pH,Nitrogen_Level,Phosphorus_Level,Potassium_Level,Chlorophyll_Content,Electrochemical_Signal]])
    pred = model.predict(X)[0]
    label = encoder.inverse_transform([pred])[0]
    return {"prediction": label}