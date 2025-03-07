import torch
import numpy as np

def load_data():
    """Carga los datasets preprocesados en formato NumPy y los convierte a tensores"""

    # Cargar imágenes
    X_train_img = torch.tensor(np.load("data/X_train_img.npy"), dtype=torch.float32)
    X_val_img = torch.tensor(np.load("data/X_val_img.npy"), dtype=torch.float32)

    # Verificar la forma de los tensores
    print("Shape original X_train_img:", X_train_img.shape)

    # Si la forma es (N, H, W, C), permutamos a (N, C, H, W)
    if X_train_img.shape[1] == 224:  # Si el segundo valor es la altura, está en formato (N, H, W, C)
        X_train_img = X_train_img.permute(0, 3, 1, 2)
        X_val_img = X_val_img.permute(0, 3, 1, 2)

    # Cargar metadata
    X_train_meta = torch.tensor(np.load("data/X_train_meta.npy"), dtype=torch.float32)
    X_val_meta = torch.tensor(np.load("data/X_val_meta.npy"), dtype=torch.float32)

    # Cargar etiquetas
    y_train = torch.tensor(np.load("data/y_train.npy"), dtype=torch.long)
    y_val = torch.tensor(np.load("data/y_val.npy"), dtype=torch.long)

    print("Shape final X_train_img:", X_train_img.shape)  # Debe ser (N, 3, 224, 224)

    return (X_train_img, X_train_meta, y_train), (X_val_img, X_val_meta, y_val)

def accuracy(preds, labels):
    """Calcula la precisión (accuracy) del modelo"""
    _, predictions = torch.max(preds, 1)
    return (predictions == labels).sum().item() / labels.size(0)