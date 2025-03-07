import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model import SkinCancerModel
from utils import load_data, accuracy

# ============================
#  CONFIGURACIÓN
# ============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 20  # 🔥 Aumentamos epochs para mayor estabilidad
LEARNING_RATE = 1e-4
PATIENCE = 5  # 🔥 Early stopping si no mejora en 5 epochs
BEST_MODEL_PATH = "models/best_model.pth"

# ============================
#  CARGA DE DATOS
# ============================

(X_train_img, X_train_meta, y_train), (X_val_img, X_val_meta, y_val) = load_data()

# Validar shapes
assert X_train_img.shape[0] == X_train_meta.shape[0] == y_train.shape[0], "❌ Error: Dimensiones en train no coinciden"
assert X_val_img.shape[0] == X_val_meta.shape[0] == y_val.shape[0], "❌ Error: Dimensiones en val no coinciden"

train_dataset = TensorDataset(X_train_img, X_train_meta, y_train)
val_dataset = TensorDataset(X_val_img, X_val_meta, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================
#  DEFINIR MODELO Y OPTIMIZADOR
# ============================

model = SkinCancerModel(num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()  # 🔥 Mixed Precision para CUDA (mejora velocidad)

# Variables para early stopping
best_val_loss = float("inf")
epochs_no_improve = 0

# ============================
#  ENTRENAMIENTO
# ============================

print(f"🚀 Entrenando en {DEVICE}...")

for epoch in range(EPOCHS):
    start_time = time.time()
    
    model.train()
    train_loss, train_acc = 0, 0

    for images, metadata, labels in train_loader:
        images, metadata, labels = images.to(DEVICE), metadata.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # 🔥 Mixed Precision Training
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_acc += accuracy(outputs, labels)

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    # Evaluación en validación
    model.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for images, metadata, labels in val_loader:
            images, metadata, labels = images.to(DEVICE), metadata.to(DEVICE), labels.to(DEVICE)

            outputs = model(images, metadata)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_acc += accuracy(outputs, labels)

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    epoch_time = time.time() - start_time
    print(f"📅 Epoch {epoch+1}/{EPOCHS} | 🕒 {epoch_time:.2f}s | 🎯 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | 🛠️ Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ============================
    #  EARLY STOPPING Y MEJOR MODELO
    # ============================

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)  # 🔥 Guardar mejor modelo
        print(f"✅ Mejor modelo guardado en {BEST_MODEL_PATH}")
    else:
        epochs_no_improve += 1
        print(f"⚠️ No mejora en {epochs_no_improve}/{PATIENCE} epochs")

    if epochs_no_improve >= PATIENCE:
        print("🛑 Early stopping activado. Finalizando entrenamiento.")
        break

# ============================
#  FINALIZACIÓN DEL ENTRENAMIENTO
# ============================

print("🎉 Entrenamiento finalizado")
print(f"✅ Mejor modelo guardado en {BEST_MODEL_PATH}")
