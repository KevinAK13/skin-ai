import pandas as pd
import logging
import json
from sklearn.preprocessing import MinMaxScaler

# =============================
#  CONFIGURACIÓN DEL PROYECTO
# =============================

CSV_PATH = "data/ISIC-images/metadata.csv"
TIMESTAMP = "20250226_222654"  # Usamos el mismo timestamp para coherencia
CLEAN_PATH = f"data/metadata_clean_{TIMESTAMP}.csv"
UNKNOWN_PATH = f"data/metadata_unknown_{TIMESTAMP}.csv"
LOG_PATH = f"data/cleaning_log_{TIMESTAMP}.log"
LABEL_MAP_PATH = "data/label_mapping.json"

# Opciones de preprocesamiento
REMOVE_DUPLICATES = True
NORMALIZE_AGE = True
ENCODE_CATEGORICAL = True

# Configuración de logs
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# =============================
#  DETECCIÓN DE VALORES ATÍPICOS
# =============================

def detect_outliers(series):
    """ Detecta y reemplaza valores atípicos usando el método IQR """
    series = series.copy()  # Evita "SettingWithCopyWarning"
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    num_outliers = ((series < lower_bound) | (series > upper_bound)).sum()
    logging.warning(f"Valores atípicos detectados en '{series.name}': {num_outliers}")

    # Reemplazar valores atípicos con la mediana
    series.loc[(series < lower_bound) | (series > upper_bound)] = series.median()
    return series

# =============================
#  FUNCIÓN PRINCIPAL PARA LIMPIEZA DEL DATASET
# =============================

def clean_dataset(csv_path=CSV_PATH, save_clean_path=CLEAN_PATH, save_unknown_path=UNKNOWN_PATH):
    """ Limpia el dataset aplicando buenas prácticas en ciencia de datos. """
    
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Iniciando limpieza del dataset: {csv_path}")
        logging.info(f"Dimensiones iniciales: {df.shape}")

        # 1️⃣ Eliminación de duplicados (opcional)
        if REMOVE_DUPLICATES:
            num_duplicates = df.duplicated().sum()
            if num_duplicates > 0:
                df = df.drop_duplicates().reset_index(drop=True)
                logging.info(f"Se eliminaron {num_duplicates} filas duplicadas.")

        # 2️⃣ Manejo de valores nulos y valores atípicos
        if "age_approx" in df.columns:
            df["age_approx"] = detect_outliers(df["age_approx"])
            df["age_approx"] = df["age_approx"].fillna(df["age_approx"].median())

        if "sex" in df.columns:
            df["sex"] = df["sex"].fillna(df["sex"].mode()[0])
            if ENCODE_CATEGORICAL:
                df["sex"] = df["sex"].map({"male": 0, "female": 1}).astype(int)

        if "anatom_site_general" in df.columns:
            df["anatom_site_general"] = df["anatom_site_general"].fillna("unknown")
            if ENCODE_CATEGORICAL:
                df["anatom_site_general"] = df["anatom_site_general"].astype("category").cat.codes

        # 3️⃣ Convertir benign/malignant en valores numéricos
        if "benign_malignant" in df.columns:
            df["benign_malignant"] = df["benign_malignant"].map({"benign": 0, "malignant": 1})

        # 4️⃣ Separar dataset `unknown` (cuando benign_malignant es nulo)
        df_unknown = df[df["benign_malignant"].isna()].reset_index(drop=True)
        df = df.dropna(subset=["benign_malignant"]).reset_index(drop=True)

        # 5️⃣ Normalización de la edad (opcional)
        if NORMALIZE_AGE and "age_approx" in df.columns and df["age_approx"].notnull().sum() > 0:
            scaler = MinMaxScaler()
            df["age_approx"] = scaler.fit_transform(df[["age_approx"]])
            if not df_unknown.empty:
                df_unknown["age_approx"] = scaler.transform(df_unknown[["age_approx"]])

        # 6️⃣ Asignación de etiquetas numéricas a "diagnosis"
        label_mapping = {label: idx for idx, label in enumerate(df["diagnosis"].unique())}
        df["label"] = df["diagnosis"].map(label_mapping)
        df_unknown["label"] = df_unknown["diagnosis"].map(label_mapping).fillna(-1).astype(int)  # -1 para etiquetas desconocidas

        with open(LABEL_MAP_PATH, "w") as f:
            json.dump(label_mapping, f)

        # 7️⃣ Eliminación de columnas irrelevantes
        cols_to_drop = ["attribution", "copyright_license", "anatom_site_special", 
                        "concomitant_biopsy", "diagnosis_1", "diagnosis_2", "diagnosis_3",
                        "diagnosis_confirm_type", "image_type", "lesion_id"]
        
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        df_unknown = df_unknown.drop(columns=[col for col in cols_to_drop if col in df_unknown.columns])

        # 8️⃣ Guardar los datasets
        df.to_csv(save_clean_path, index=False)
        df_unknown.to_csv(save_unknown_path, index=False)

        logging.info(f"Dataset limpio guardado con {df.shape[0]} filas.")
        logging.info(f"Dataset 'unknown' guardado con {df_unknown.shape[0]} filas.")

        return df, df_unknown

    except Exception as e:
        logging.error(f"Error en la limpieza de datos: {str(e)}")
        return None, None

# =============================
#  EJECUCIÓN DEL SCRIPT
# =============================

df_clean, df_unknown = clean_dataset()

# =============================
#  VERIFICACIÓN DEL DATASET
# =============================

def plot_class_distribution(df, column="diagnosis", title="Distribución de Clases en Dataset Limpio"):
    """ Genera un gráfico de barras normalizado para visualizar la distribución de clases después de la limpieza. """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(8,5))
    ax = sns.barplot(x=df[column].value_counts(normalize=True).index, 
                     y=df[column].value_counts(normalize=True).values)
    ax.bar_label(ax.containers[0], fmt="%.2f")
    plt.xlabel("Diagnóstico")
    plt.ylabel("Proporción")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

if df_clean is not None:
    plot_class_distribution(df_clean)