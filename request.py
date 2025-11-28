import requests

url = "http://localhost:5000/predict"
data = {
    "Soil_Moisture": 30,
    "Ambient_Temperature": 24,
    "Soil_Temperature": 20,
    "Humidity": 65,
    "Light_Intensity": 5500,
    "Soil_pH": 6.5,
    "Nitrogen_Level": 40,
    "Phosphorus_Level": 25,
    "Potassium_Level": 180,
    "Chlorophyll_Content": 52
}

response = requests.post(url, json=data)
print(response.json())
