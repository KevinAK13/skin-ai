# Usa una imagen base ligera
FROM python:3.10-slim

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia los archivos necesarios
COPY requirements.txt .
COPY backend /app/backend
COPY models /app/models

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto de FastAPI
EXPOSE 8000

# Ejecuta Uvicorn asegurando la ruta correcta
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]