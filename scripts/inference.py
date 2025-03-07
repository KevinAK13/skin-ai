import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import joblib
from model import SkinCancerModel  # Asegúrate de importar el modelo

# ============================
#  CONFIGURACIÓN
# ============================

MODEL_PATH = "models/best_model.pth"
SCALER_PATH = "models/scaler.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (224, 224)

# ============================
#  CARGAR MODELO Y ESCALADOR
# ============================

model = SkinCancerModel(num_classes=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()  # Modo evaluación

scaler = joblib.load(SCALER_PATH)  # Cargar el escalador de edad y sexo

# ============================
#  TRANSFORMACIÓN DE IMÁGENES
# ============================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image_path, age, sex):
    """
    Realiza una predicción con el modelo de detección de cáncer de piel.
    
    Parámetros:
        image_path (str): Ruta de la imagen.
        age (int): Edad de la persona.
        sex (str): Género ("male" o "female").
    
    Retorna:
        dict: Resultado de la predicción.
    """

    # Cargar imagen
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0).to(DEVICE)  # Añadir batch dimension

    # Preprocesar edad y sexo
    sex_num = 0 if sex.lower() == "male" else 1
    metadata = np.array([[age, sex_num]], dtype=np.float32)
    metadata = scaler.transform(metadata)  # Aplicar escalador
    metadata = torch.tensor(metadata, dtype=torch.float32).to(DEVICE)

    # Hacer predicción
    with torch.no_grad():
        output = model(img, metadata)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities).item()
    
    # Interpretar resultado
    diagnosis = "Maligno (melanoma)" if predicted_class == 1 else "Benigno"
    confidence = probabilities[0, predicted_class].item()

    return {
        "diagnosis": diagnosis,
        "confidence": confidence
    }

# ============================
#  EJEMPLO DE PREDICCIÓN
# ============================

if __name__ == "__main__":
    image_path = "data/ISIC-images/ISIC_0024345.jpg"  # Ruta de la imagen de prueba
    age = 45
    sex = "male"

    result = predict(image_path, age, sex)
    print(f"🔍 Diagnóstico: {result['diagnosis']} (Confianza: {result['confidence']:.2%})")