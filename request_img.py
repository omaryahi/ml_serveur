import requests

url = "http://localhost:5000/predict/plant3"
files = {"file": open("test_image.jpg", "rb")}

response = requests.post(url, files=files)
print(response.json())
