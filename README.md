# Skin Cancer Classification Model Documentation

## Overview

This documentation provides a comprehensive guide to the Skin Cancer Classification Model, which combines a Convolutional Neural Network (CNN) based on EfficientNet with a Multilayer Perceptron (MLP) to classify skin lesions as benign or malignant. The model processes both image data (skin lesion images) and tabular data (age and sex) to improve classification accuracy. This documentation is designed to be accessible to web developers, Python programmers, medical professionals, researchers, and anyone interested in Machine Learning and Artificial Intelligence.

## Table of Contents

1. **Introduction**
2. **Model Architecture**
3. **Data Preprocessing**
4. **Training Process**
5. **API Explanation**
6. **Ethical Considerations**
7. **Conclusion**

---

## 1. Introduction

The Skin Cancer Classification Model is designed to assist in the early detection of skin cancer by classifying skin lesions as either benign or malignant. The model leverages a combination of image data (dermatoscopic images) and tabular data (patient age and sex) to make more accurate predictions. This approach aims to improve the diagnostic process, providing a tool that can be used by medical professionals to support their clinical decisions.

The model is built using PyTorch and incorporates EfficientNet-B0 for image feature extraction and an MLP for processing tabular data. The combination of these two components allows the model to capture both visual and demographic information, leading to a more robust classification.

---

## 2. Model Architecture

### 2.1 EfficientNet-B0 for Image Feature Extraction

EfficientNet-B0 is a lightweight and efficient CNN architecture that is pre-trained on the ImageNet dataset. It is used to extract features from dermatoscopic images of skin lesions. The final classification layer of EfficientNet is removed, and the model outputs a feature vector that represents the visual characteristics of the lesion.

### 2.2 MLP for Tabular Data Processing

The MLP processes tabular data, specifically the patient's age and sex. It consists of two fully connected layers with ReLU activation functions. The output of the MLP is a feature vector that captures demographic information relevant to the classification task.

### 2.3 Combined Classification

The feature vectors from EfficientNet and the MLP are concatenated and passed through a final classification layer. This layer consists of a fully connected network with 256 neurons, followed by a dropout layer to prevent overfitting, and a final output layer that provides the classification probabilities (benign or malignant).

```python
class SkinCancerModel(nn.Module):
    def __init__(self, num_classes=2):
        super(SkinCancerModel, self).__init__()
        
        # CNN for images (EfficientNet)
        self.cnn = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.cnn.classifier = nn.Identity()  # Remove the final classification layer

        # MLP for tabular data (age and sex)
        self.mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        # Calculate the output size of EfficientNet
        dummy_input = torch.randn(1, 3, 224, 224)
        img_output_size = self.cnn(dummy_input).shape[-1]

        # Final combined classification layer
        self.classifier = nn.Sequential(
            nn.Linear(img_output_size + 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, metadata):
        img_features = self.cnn(image)
        meta_features = self.mlp(metadata)
        combined = torch.cat((img_features, meta_features), dim=1)
        output = self.classifier(combined)
        return output
```

---

## 3. Data Preprocessing

### 3.1 Data Cleaning

The dataset used for training the model is the HAM10000 dataset, which contains 11,720 dermatoscopic images of common pigmented skin lesions. The dataset is preprocessed to ensure high-quality input data for the model. This includes:

- **Handling Missing Values**: Missing values in the dataset are imputed using appropriate strategies (e.g., median for age, mode for sex).
- **Normalization**: Numerical data (age) is normalized using Min-Max Scaling to ensure consistent input ranges.
- **Encoding Categorical Variables**: Categorical variables (e.g., sex) are encoded into numerical values.
- **Removing Irrelevant Columns**: Columns that do not contribute to the model (e.g., lesion ID, image type) are removed.

### 3.2 Data Augmentation

To improve the model's robustness, data augmentation techniques are applied to the images. These include:

- **Rotation and Flipping**: Images are randomly rotated and flipped to simulate different orientations.
- **Brightness and Contrast Adjustment**: Random adjustments to brightness and contrast are made to simulate different lighting conditions.
- **Gaussian Blur and Elastic Transform**: These transformations simulate blurry or distorted images, helping the model generalize better.

```python
transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=20, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.CoarseDropout(num_holes=1, max_holes=2, max_height=20, max_width=20, p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

---

## 4. Training Process

### 4.1 Dataset Splitting

The dataset is split into training (70%), validation (15%), and test (15%) sets. This ensures that the model is evaluated on unseen data, providing a more accurate measure of its performance.

### 4.2 Model Training

The model is trained using the Adam optimizer with a learning rate of 1e-4. Early stopping is implemented to prevent overfitting, where training is halted if the validation loss does not improve for a specified number of epochs (patience=5).

```python
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    
    for images, metadata, labels in train_loader:
        images, metadata, labels = images.to(DEVICE), metadata.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images, metadata)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for images, metadata, labels in val_loader:
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("🛑 Early stopping activated")
            break
```

---

## 5. API Explanation

The model is accessible via an API that allows users to submit images and metadata for classification. The API processes the input data, applies the necessary transformations, and returns the classification result (benign or malignant).

### 5.1 API Endpoints

- **POST /classify**: Accepts an image and metadata (age, sex) and returns the classification result.

### 5.2 Example Request

```json
{
    "image": "base64_encoded_image",
    "age": 45,
    "sex": "male"
}
```

### 5.3 Example Response

```json
{
    "classification": "benign",
    "confidence": 0.92
}
```

---

## 6. Ethical Considerations

The development and deployment of this model adhere to strict ethical guidelines:

- **Data Privacy**: Patient data is anonymized, and no personally identifiable information is stored or shared.
- **Bias Mitigation**: The dataset is carefully curated to ensure diversity in age, sex, and skin types, reducing the risk of bias in the model's predictions.
- **Transparency**: All preprocessing steps, model architecture, and training procedures are documented to ensure reproducibility and transparency.
- **Informed Consent**: The use of patient data complies with ethical standards, and appropriate consent is obtained where required.

---

## 7. Conclusion

The Skin Cancer Classification Model is a powerful tool that combines image and tabular data to improve the accuracy of skin lesion classification. By leveraging EfficientNet for image feature extraction and an MLP for processing demographic data, the model provides a robust and reliable classification system. The model is designed with ethical considerations in mind, ensuring that it can be used responsibly in clinical settings.

This documentation provides a comprehensive guide to the model's architecture, data preprocessing, training process, and API usage, making it accessible to a wide range of stakeholders, including developers, researchers, and medical professionals.

---

**Note**: This documentation is intended to be a living document. As the model evolves, updates will be made to reflect new features, improvements, and changes in best practices.



# Dokumentation des Hautkrebs-Klassifikationsmodells

## Überblick

Diese Dokumentation bietet einen umfassenden Leitfaden für das Hautkrebs-Klassifikationsmodell, das ein Convolutional Neural Network (CNN) basierend auf EfficientNet mit einem Multilayer Perceptron (MLP) kombiniert, um Hautläsionen als gutartig oder bösartig zu klassifizieren. Das Modell verarbeitet sowohl Bilddaten (dermatoskopische Bilder) als auch tabellarische Daten (Alter und Geschlecht), um die Klassifikationsgenauigkeit zu verbessern. Diese Dokumentation ist so gestaltet, dass sie für Webentwickler, Python-Programmierer, medizinisches Personal, Forscher und alle, die an maschinellem Lernen und künstlicher Intelligenz interessiert sind, zugänglich ist.

## Inhaltsverzeichnis

1. **Einführung**
2. **Modellarchitektur**
3. **Datenvorverarbeitung**
4. **Trainingsprozess**
5. **API-Erklärung**
6. **Ethische Überlegungen**
7. **Fazit**

---

## 1. Einführung

Das Hautkrebs-Klassifikationsmodell wurde entwickelt, um die Früherkennung von Hautkrebs zu unterstützen, indem es Hautläsionen als gutartig oder bösartig klassifiziert. Das Modell nutzt eine Kombination aus Bilddaten (dermatoskopische Bilder) und tabellarischen Daten (Alter und Geschlecht des Patienten), um genauere Vorhersagen zu treffen. Dieser Ansatz zielt darauf ab, den diagnostischen Prozess zu verbessern und medizinischem Fachpersonal ein Werkzeug an die Hand zu geben, das ihre klinischen Entscheidungen unterstützt.

Das Modell wurde mit PyTorch entwickelt und verwendet EfficientNet-B0 zur Extraktion von Bildmerkmalen sowie ein MLP zur Verarbeitung tabellarischer Daten. Die Kombination dieser beiden Komponenten ermöglicht es dem Modell, sowohl visuelle als auch demografische Informationen zu erfassen, was zu einer robusteren Klassifikation führt.

---

## 2. Modellarchitektur

### 2.1 EfficientNet-B0 zur Extraktion von Bildmerkmalen

EfficientNet-B0 ist eine leichte und effiziente CNN-Architektur, die auf dem ImageNet-Datensatz vortrainiert wurde. Es wird verwendet, um Merkmale aus dermatoskopischen Bildern von Hautläsionen zu extrahieren. Die letzte Klassifikationsschicht von EfficientNet wird entfernt, und das Modell gibt einen Feature-Vektor aus, der die visuellen Eigenschaften der Läsion repräsentiert.

### 2.2 MLP zur Verarbeitung tabellarischer Daten

Das MLP verarbeitet tabellarische Daten, insbesondere das Alter und das Geschlecht des Patienten. Es besteht aus zwei vollständig verbundenen Schichten mit ReLU-Aktivierungsfunktionen. Die Ausgabe des MLP ist ein Feature-Vektor, der demografische Informationen erfasst, die für die Klassifikation relevant sind.

### 2.3 Kombinierte Klassifikation

Die Feature-Vektoren von EfficientNet und dem MLP werden zusammengeführt und durch eine finale Klassifikationsschicht geleitet. Diese Schicht besteht aus einem vollständig verbundenen Netzwerk mit 256 Neuronen, gefolgt von einer Dropout-Schicht zur Vermeidung von Overfitting, und einer finalen Ausgabeschicht, die die Klassifikationswahrscheinlichkeiten (gutartig oder bösartig) liefert.

```python
class SkinCancerModel(nn.Module):
    def __init__(self, num_classes=2):
        super(SkinCancerModel, self).__init__()
        
        # CNN für Bilder (EfficientNet)
        self.cnn = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.cnn.classifier = nn.Identity()  # Entfernt die finale Klassifikationsschicht

        # MLP für tabellarische Daten (Alter und Geschlecht)
        self.mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        # Berechnet die Ausgabegröße von EfficientNet
        dummy_input = torch.randn(1, 3, 224, 224)
        img_output_size = self.cnn(dummy_input).shape[-1]

        # Finale kombinierte Klassifikationsschicht
        self.classifier = nn.Sequential(
            nn.Linear(img_output_size + 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, metadata):
        img_features = self.cnn(image)
        meta_features = self.mlp(metadata)
        combined = torch.cat((img_features, meta_features), dim=1)
        output = self.classifier(combined)
        return output
```

---

## 3. Datenvorverarbeitung

### 3.1 Datenbereinigung

Der für das Training des Modells verwendete Datensatz ist der HAM10000-Datensatz, der 11.720 dermatoskopische Bilder von häufigen pigmentierten Hautläsionen enthält. Der Datensatz wird vorverarbeitet, um hochwertige Eingabedaten für das Modell sicherzustellen. Dies umfasst:

- **Handhabung fehlender Werte**: Fehlende Werte im Datensatz werden mit geeigneten Strategien ersetzt (z.B. Median für das Alter, Modus für das Geschlecht).
- **Normalisierung**: Numerische Daten (Alter) werden mit Min-Max-Scaling normalisiert, um konsistente Eingabebereiche sicherzustellen.
- **Kodierung kategorischer Variablen**: Kategorische Variablen (z.B. Geschlecht) werden in numerische Werte umgewandelt.
- **Entfernen irrelevanter Spalten**: Spalten, die nicht zum Modell beitragen (z.B. Läsions-ID, Bildtyp), werden entfernt.

### 3.2 Datenaugmentierung

Um die Robustheit des Modells zu verbessern, werden Datenaugmentierungstechniken auf die Bilder angewendet. Dazu gehören:

- **Rotation und Spiegelung**: Bilder werden zufällig gedreht und gespiegelt, um verschiedene Ausrichtungen zu simulieren.
- **Helligkeits- und Kontrastanpassung**: Zufällige Anpassungen von Helligkeit und Kontrast simulieren unterschiedliche Lichtverhältnisse.
- **Gaußscher Weichzeichner und elastische Transformation**: Diese Transformationen simulieren unscharfe oder verzerrte Bilder und helfen dem Modell, besser zu generalisieren.

```python
transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=20, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.CoarseDropout(num_holes=1, max_holes=2, max_height=20, max_width=20, p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

---

## 4. Trainingsprozess

### 4.1 Aufteilung des Datensatzes

Der Datensatz wird in Trainings- (70%), Validierungs- (15%) und Testdaten (15%) aufgeteilt. Dies stellt sicher, dass das Modell anhand von unbekannten Daten evaluiert wird, was eine genauere Messung seiner Leistung ermöglicht.

### 4.2 Modelltraining

Das Modell wird mit dem Adam-Optimizer und einer Lernrate von 1e-4 trainiert. Early Stopping wird implementiert, um Overfitting zu verhindern, wobei das Training abgebrochen wird, wenn sich der Validierungsverlust über eine bestimmte Anzahl von Epochen (Patience=5) nicht verbessert.

```python
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    
    for images, metadata, labels in train_loader:
        images, metadata, labels = images.to(DEVICE), metadata.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images, metadata)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for images, metadata, labels in val_loader:
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("🛑 Early stopping aktiviert")
            break
```

---

## 5. API-Erklärung

Das Modell ist über eine API zugänglich, die es Benutzern ermöglicht, Bilder und Metadaten zur Klassifikation einzureichen. Die API verarbeitet die Eingabedaten, wendet die notwendigen Transformationen an und gibt das Klassifikationsergebnis (gutartig oder bösartig) zurück.

### 5.1 API-Endpunkte

- **POST /classify**: Akzeptiert ein Bild und Metadaten (Alter, Geschlecht) und gibt das Klassifikationsergebnis zurück.

### 5.2 Beispielanfrage

```json
{
    "image": "base64_encoded_image",
    "age": 45,
    "sex": "male"
}
```

### 5.3 Beispielantwort

```json
{
    "classification": "benign",
    "confidence": 0.92
}
```

---

## 6. Ethische Überlegungen

Die Entwicklung und Bereitstellung dieses Modells folgt strengen ethischen Richtlinien:

- **Datenschutz**: Patientendaten werden anonymisiert, und keine personenbezogenen Informationen werden gespeichert oder weitergegeben.
- **Bias-Minderung**: Der Datensatz wird sorgfältig kuratiert, um Diversität in Alter, Geschlecht und Hauttypen sicherzustellen, wodurch das Risiko von Verzerrungen in den Vorhersagen des Modells reduziert wird.
- **Transparenz**: Alle Vorverarbeitungsschritte, die Modellarchitektur und die Trainingsverfahren sind dokumentiert, um Reproduzierbarkeit und Transparenz zu gewährleisten.
- **Informierte Zustimmung**: Die Verwendung von Patientendaten entspricht ethischen Standards, und wo erforderlich, wird eine entsprechende Zustimmung eingeholt.

---

## 7. Fazit

Das Hautkrebs-Klassifikationsmodell ist ein leistungsstarkes Werkzeug, das Bild- und tabellarische Daten kombiniert, um die Genauigkeit der Klassifikation von Hautläsionen zu verbessern. Durch die Nutzung von EfficientNet zur Extraktion von Bildmerkmalen und eines MLP zur Verarbeitung demografischer Daten bietet das Modell ein robustes und zuverlässiges Klassifikationssystem. Das Modell wurde unter Berücksichtigung ethischer Aspekte entwickelt, um sicherzustellen, dass es verantwortungsvoll in klinischen Umgebungen eingesetzt werden kann.

Diese Dokumentation bietet einen umfassenden Leitfaden zur Modellarchitektur, Datenvorverarbeitung, Trainingsprozess und API-Nutzung, wodurch sie für eine breite Palette von Interessengruppen, einschließlich Entwicklern, Forschern und medizinischem Fachpersonal, zugänglich ist.

---

**Hinweis**: Diese Dokumentation ist als lebendes Dokument gedacht. Wenn sich das Modell weiterentwickelt, werden Aktualisierungen vorgenommen, um neue Funktionen, Verbesserungen und Änderungen in den Best Practices widerzuspiegeln.