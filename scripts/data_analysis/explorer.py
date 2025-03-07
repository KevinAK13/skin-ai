import pandas as pd

# 📌 Ruta del dataset
CSV_PATH = "data/ISIC-images/metadata.csv"

# 📌 Cargar dataset
df = pd.read_csv(CSV_PATH)

# 📌 Ver las primeras filas del dataset
print("🔍 Primeras filas del CSV:")
print(df.head())

# 📌 Ver columnas disponibles
print("\n📌 Columnas en el dataset:")
print(df.columns.tolist())

# 📌 Ver dimensiones del dataset
print("\n📌 Dimensiones del dataset:", df.shape)

# 📌 Ver si hay valores nulos
print("\n📌 Valores nulos en el dataset:")
print(df.isnull().sum())

# 📌 Ver cuántos registros hay por cada tipo de diagnóstico
print("\n📌 Distribución de diagnósticos:")
print(df["diagnosis"].value_counts(normalize=True))