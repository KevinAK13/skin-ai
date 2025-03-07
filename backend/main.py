import torch
import uvicorn
import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms
from backend.model import SkinCancerModel
import os
from fastapi.middleware.cors import CORSMiddleware


# ============================
#  CONFIGURATION / KONFIGURATION
# ============================

app = FastAPI(
    title="Skin Cancer Detection API",  # API for skin cancer detection / API zur Hautkrebserkennung
    description="API for skin cancer detection using a Machine Learning model based on images of skin lesions. / API zur Hautkrebserkennung mit einem Machine-Learning-Modell basierend auf Bildern von Hautläsionen.",
    version="1.1",
    contact={
        "name": "Kevin Guerra Xochihua",
        "url": "https://gitlab.com/kevin.guerra",
        "email": "k.guerraxochihua@gmail.com",
    },
)

# Configure CORS / CORS konfigurieren
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the current directory path / Hole den Pfad des aktuellen Verzeichnisses
MODEL_PATH = os.path.join(BASE_DIR, "../models/best_model.pth")  # Absolute path to the model / Absoluter Pfad zum Modell
SCALER_PATH = os.path.join(BASE_DIR, "../models/scaler.pkl")  # Absolute path to the scaler / Absoluter Pfad zum Scaler
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available / Verwende GPU, falls verfügbar
IMAGE_SIZE = (224, 224)

# ============================
#  LOAD MODEL AND SCALER WITH ERROR HANDLING / MODELL UND SCALER MIT FEHLERBEHANDLUNG LADEN
# ============================

try:
    model = SkinCancerModel(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
except Exception as e:
    raise RuntimeError(f"Error loading the model: {str(e)} / Fehler beim Laden des Modells: {str(e)}")

try:
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading the scaler: {str(e)} / Fehler beim Laden des Scalers: {str(e)}")

# ============================
#  IMAGE TRANSFORMATION / BILDTRANSFORMATION
# ============================

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============================
#  RESPONSE MODEL / ANTWORTMODELL
# ============================

class PredictionResponse(BaseModel):
    diagnosis: str  # Diagnosis / Diagnose
    confidence: float  # Confidence / Konfidenz

# Global variable to store the last prediction / Globale Variable zur Speicherung der letzten Vorhersage
last_prediction = {}

# ============================
#  INFERENCE ENDPOINT / INFERENZ-ENDPOINT
# ============================

@app.post("/predict/", response_model=PredictionResponse, summary="Perform skin cancer diagnosis / Hautkrebsdiagnose durchführen")
async def predict(
    file: UploadFile = File(..., description="Image of the skin lesion (JPG/PNG format) / Bild der Hautläsion (JPG/PNG-Format)"),
    age: int = Form(..., description="Patient's age (years) / Alter des Patienten (Jahre)"),
    sex: str = Form(..., description="Patient's sex ('male' or 'female') / Geschlecht des Patienten ('male' oder 'female')")
):
    """
     **Perform a skin cancer prediction based on the image, age, and sex. / Führe eine Hautkrebsvorhersage basierend auf Bild, Alter und Geschlecht durch.**

     **Parameters: / Parameter:**
    - `file`:  Image of the skin lesion. / Bild der Hautläsion.
    - `age`:  Patient's age. / Alter des Patienten.
    - `sex`:  Patient's sex ('male' or 'female'). / Geschlecht des Patienten ('male' oder 'female').

     **Returns: / Gibt zurück:**
    - `diagnosis`:  Diagnosis (`Benign` or `Malignant (melanoma)`). / Diagnose (`Gutartig` oder `Bösartig (Melanom)`).
    - `confidence`:  Model's confidence in the diagnosis (percentage). / Konfidenz des Modells in der Diagnose (Prozent).
    """
    # Image validation / Bildvalidierung
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported image format. Use JPG or PNG. / Nicht unterstütztes Bildformat. Verwende JPG oder PNG.")

    try:
        # Process image / Bild verarbeiten
        img = Image.open(file.file).convert("RGB")
        img = transform(img).unsqueeze(0).to(DEVICE)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image. / Ungültiges oder beschädigtes Bild.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)} / Fehler bei der Bildverarbeitung: {str(e)}")

    # Age validation / Altersvalidierung
    if age < 0 or age > 120:
        raise HTTPException(status_code=400, detail="Age must be between 0 and 120 years. / Das Alter muss zwischen 0 und 120 Jahren liegen.")

    # Sex validation / Geschlechtsvalidierung
    if sex.lower() not in ["male", "female"]:
        raise HTTPException(status_code=400, detail="Sex must be 'male' or 'female'. / Das Geschlecht muss 'male' oder 'female' sein.")

    try:
        # Normalize age and sex / Alter und Geschlecht normalisieren
        sex_num = 0 if sex.lower() == "male" else 1
        metadata = np.array([[age, sex_num]], dtype=np.float32)
        metadata = scaler.transform(metadata)
        metadata = torch.tensor(metadata, dtype=torch.float32).to(DEVICE)

        # Inference / Inferenz
        with torch.no_grad():
            output = model(img, metadata)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities).item()

        # Result / Ergebnis
        diagnosis = "Malignant (melanoma)" if predicted_class == 1 else "Benign"
        confidence = probabilities[0, predicted_class].item()

        # Save the last prediction / Letzte Vorhersage speichern
        global last_prediction
        last_prediction = {
            "age": age,
            "sex": sex,
            "diagnosis": diagnosis,
            "confidence": round(confidence * 100, 2),
        }

        return PredictionResponse(
            diagnosis=diagnosis,
            confidence=round(confidence * 100, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model inference: {str(e)} / Fehler während der Modellinferenz: {str(e)}")

# ============================
#  ENDPOINT TO GET THE LAST RESULT / ENDPOINT ZUM ABRUFEN DES LETZTEN ERGEBNISSES
# ============================

@app.get("/result/", summary="Get the last prediction result / Hole das letzte Vorhersageergebnis")
async def get_last_result():
    """
     **Returns the last prediction made. / Gibt die letzte durchgeführte Vorhersage zurück.**

     **Returns: / Gibt zurück:**
    - `age`: Patient's age. / Alter des Patienten.
    - `sex`: Patient's sex. / Geschlecht des Patienten.
    - `diagnosis`: Diagnosis (`Benign` or `Malignant (melanoma)`). / Diagnose (`Gutartig` oder `Bösartig (Melanom)`).
    - `confidence`: Model's confidence in the diagnosis (percentage). / Konfidenz des Modells in der Diagnose (Prozent).
    """
    if not last_prediction:
        raise HTTPException(status_code=404, detail="No predictions stored yet. / Noch keine Vorhersagen gespeichert.")
    
    return last_prediction

# ============================
#  HEALTH CHECK ENDPOINT / HEALTH-CHECK-ENDPOINT
# ============================

@app.get("/health", summary="Check the API status / Überprüfe den API-Status")
async def health_check():
    return {"status": "ok", "device": DEVICE}

# ============================
#  GLOBAL ERROR HANDLING / GLOBALE FEHLERBEHANDLUNG
# ============================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return {"error": "Internal server error / Interner Serverfehler", "detail": str(exc)}

# ============================
#  START SERVER / SERVER STARTEN
# ============================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)