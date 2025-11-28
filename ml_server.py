from fastapi import FastAPI ,UploadFile, File
from torchvision import models
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from pydantic import BaseModel
import pandas as pd
import joblib
import torch.nn as nn

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
@app.post("/predict/env")
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

#---------------------------------------------------------------------------
models_dir = "models_img"
loaded_models = {}

# -------------------------------------------------------
# IMAGE TRANSFORMS (same as training!)
# -------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------------------------------------
# LOAD ALL MODELS (apple.pth, potato.pth, orange.pth...)
# -------------------------------------------------------
for file in os.listdir(models_dir):
    if not file.endswith(".pth"):
        continue

    plant_name = file.replace(".pth", "").lower()
    pth_path = os.path.join(models_dir, file)

    print(f"Loading: {plant_name}")

    # Load state dict (safe)
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=False)

    # Detect number of classes
    fc_weight = state_dict["fc.weight"]
    num_classes = fc_weight.shape[0]
    print(f" → Detected {num_classes} classes")

    # Create empty resnet18 with correct head
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load weights
    model.load_state_dict(state_dict)
    model.eval()

    loaded_models[plant_name] = model

print("All models loaded:", list(loaded_models.keys()))


# -------------------------------------------------------
# Helper: predict with selected model
# -------------------------------------------------------
def predict_image(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()


# -------------------------------------------------------
# DYNAMIC ENDPOINTS FOR EACH PLANT
# -------------------------------------------------------
@app.post("/predict/{plant_name}")
async def predict_plant(plant_name: str, file: UploadFile = File(...)):
    plant_name = plant_name.lower()

    if plant_name not in loaded_models:
        raise HTTPException(status_code=404, detail="Plant model not found.")

    model = loaded_models[plant_name]

    # Load image
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Preprocess
    tensor = transform(image).unsqueeze(0)

    # Predict
    class_id = predict_image(model, tensor)

    return {
        "plant": plant_name,
        "predicted_class": int(class_id)
    }