import pandas as pd

# Cargar el dataset limpio
CSV_PATH = "data/ISIC-images/metadata.csv"  # Ruta del archivo CSV
df = pd.read_csv(CSV_PATH)

# Ver estadísticas de la edad
media_edad = df["age_approx"].mean()
std_edad = df["age_approx"].std()
min_edad = df["age_approx"].min()
max_edad = df["age_approx"].max()

print(f"📊 Estadísticas de edad en el dataset:")
print(f"🔹 Media: {media_edad:.2f}")
print(f"🔹 Desviación estándar: {std_edad:.2f}")
print(f"🔹 Edad mínima: {min_edad}")
print(f"🔹 Edad máxima: {max_edad}")


from sklearn.preprocessing import StandardScaler

# Extraer edades
edades = df[["age_approx"]].values  # Necesita estar en 2D para StandardScaler

# Aplicar StandardScaler
scaler = StandardScaler()
edades_normalizadas = scaler.fit_transform(edades)

# Agregar la edad normalizada al DataFrame
df["age_normalized"] = edades_normalizadas

# Mostrar algunas filas
print(df[["age_approx", "age_normalized"]].head(10))