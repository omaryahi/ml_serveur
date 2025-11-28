from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load your model and label encoder
model = joblib.load("plant_health_model_xgb.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define input schema
class PlantData(BaseModel):
    Soil_Moisture: float
    Ambient_Temperature: float
    Soil_Temperature: float
    Humidity: float
    Light_Intensity: float
    Soil_pH: float
    Nitrogen_Level: float
    Phosphorus_Level: float
    Potassium_Level: float
    Chlorophyll_Content: float
    Electrochemical_Signal: float = -140  # default value

# Prediction endpoint
@app.post("/predict")
def predict_plant_health(data: PlantData):
    # Convert to DataFrame
    df = pd.DataFrame([[
        data.Soil_Moisture,
        data.Ambient_Temperature,
        data.Soil_Temperature,
        data.Humidity,
        data.Light_Intensity,
        data.Soil_pH,
        data.Nitrogen_Level,
        data.Phosphorus_Level,
        data.Potassium_Level,
        data.Chlorophyll_Content,
        data.Electrochemical_Signal
    ]], columns=[
        "Soil_Moisture",
        "Ambient_Temperature",
        "Soil_Temperature",
        "Humidity",
        "Light_Intensity",
        "Soil_pH",
        "Nitrogen_Level",
        "Phosphorus_Level",
        "Potassium_Level",
        "Chlorophyll_Content",
        "Electrochemical_Signal"
    ])

    # Predict
    pred_encoded = model.predict(df)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    return {"prediction": pred_label}
