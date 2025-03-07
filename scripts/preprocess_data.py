import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # Para guardar el scaler
from tqdm import tqdm

# ============================
#  CONFIGURACIÓN
# ============================

DATASET_PATH = "data/ISIC-images"
CSV_PATH = "data/metadata_clean_20250226_222654.csv"
IMAGE_SIZE = (224, 224)

# ============================
#  CARGAR METADATA LIMPIA
# ============================

df = pd.read_csv(CSV_PATH)
df["img_path"] = df["isic_id"].apply(lambda x: os.path.join(DATASET_PATH, f"{x}.jpg"))
df = df[df["img_path"].apply(os.path.exists)].reset_index(drop=True)

# ============================
#  DEFINIR TRANSFORMACIONES
# ============================

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

transform = A.Compose([
    A.Resize(*IMAGE_SIZE),  # Redimensionar a 224x224
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=20, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),  # Simula desenfoque de piel
    A.CoarseDropout(
        num_holes_range=(1, 2),  # Rango de número de regiones a eliminar
        hole_height_range=(10, 20),  # Rango de altura de eliminación
        hole_width_range=(10, 20),  # Rango de ancho de eliminación
        fill=0,  # Relleno negro (puede ser "random_uniform" para variabilidad)
        p=0.3
    ),  # Simula oclusión
    A.ElasticTransform(alpha=1, sigma=50, p=0.2),  # Simula deformaciones en la piel
    A.Normalize(mean=imagenet_mean, std=imagenet_std, max_pixel_value=255.0),
    ToTensorV2()
])

images, labels, ages, sexes = [], [], [], []

# ============================
#  PROCESAR IMÁGENES
# ============================

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_path = row["img_path"]
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(image=img)["image"].numpy()

        images.append(img)
        labels.append(row["label"])
        ages.append(row["age_approx"])
        sexes.append(row["sex"])

# ============================
#  NORMALIZACIÓN DE METADATOS
# ============================

X_images = np.array(images, dtype=np.float32)
X_metadata = np.array(list(zip(ages, sexes)), dtype=np.float32)

scaler = StandardScaler()
X_metadata = scaler.fit_transform(X_metadata)
joblib.dump(scaler, "models/scaler.pkl")

y = np.array(labels)

# ============================
#  DIVIDIR EN TRAIN/VAL/TEST
# ============================

X_train_img, X_temp_img, X_train_meta, X_temp_meta, y_train, y_temp = train_test_split(
    X_images, X_metadata, y, test_size=0.3, random_state=42, stratify=y)

X_val_img, X_test_img, X_val_meta, X_test_meta, y_val, y_test = train_test_split(
    X_temp_img, X_temp_meta, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# ============================
#  GUARDAR LOS DATASETS
# ============================

np.save("data/X_train_img.npy", X_train_img)
np.save("data/X_train_meta.npy", X_train_meta)
np.save("data/y_train.npy", y_train)

np.save("data/X_val_img.npy", X_val_img)
np.save("data/X_val_meta.npy", X_val_meta)
np.save("data/y_val.npy", y_val)

np.save("data/X_test_img.npy", X_test_img)
np.save("data/X_test_meta.npy", X_test_meta)
np.save("data/y_test.npy", y_test)

print(f"✅ Dataset procesado: Training {X_train_img.shape[0]}, Validation {X_val_img.shape[0]}, Test {X_test_img.shape[0]}")