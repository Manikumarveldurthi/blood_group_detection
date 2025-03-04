from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from .model import BloodGroupCNN, predict_single_image, get_grad_cam, advanced_xai_plots
import os
from datetime import datetime
import random
import torch.nn as nn

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_labels = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]

model = BloodGroupCNN(len(class_labels))
model.load_state_dict(torch.load("./models/blood_group_model.pth", map_location=device))
model = model.to(device)  
model.eval()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(
    name: str = Form(...),
    age: int = Form(...),
    address: str = Form(...),
    fingerprint: UploadFile = File(...)
):
    temp_path = f"temp_{fingerprint.filename}"
    with open(temp_path, "wb") as buffer:
        content = await fingerprint.read()
        buffer.write(content)
    
    with torch.no_grad():
        prediction_result, image_tensor, class_idx = predict_single_image(model, temp_path)
    
    heatmap = get_grad_cam(model, image_tensor, class_idx)
    
    plt.switch_backend('Agg')
    plots = advanced_xai_plots(temp_path, heatmap)
    
    plots_base64 = {}
    for plot_name, plot_data in plots.items():
        buffered = BytesIO()
        plot_data.savefig(buffered, format="png")
        plots_base64[plot_name] = base64.b64encode(buffered.getvalue()).decode()
        plt.close(plot_data)
    
    with open(temp_path, "rb") as image_file:
        fingerprint_base64 = base64.b64encode(image_file.read()).decode()
    
    os.remove(temp_path)
    
    return {
        "name": name,
        "age": age,
        "address": address,
        "predicted_blood_type": prediction_result["blood_type"],
        "confidence": prediction_result["confidence"],
        "fingerprint_image": fingerprint_base64,
        "plots": plots_base64,
        "report_date": datetime.now().strftime("%d %B, %Y"),
        "report_id": f"BT{random.randint(10000,99999)}"
    } 