# =======================================
# PUNTO 1 - MODELO DE APRENDIZAJE CLÁSICO
# =======================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# 1. Cargar dataset
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "train.csv")

df = pd.read_csv(DATA_PATH)

print(" Dataset cargado con éxito")
print(df.head(), "\n")

# 2. Preprocesamiento
print("🔧 Preprocesando datos...")
# Eliminar columnas irrelevantes
df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Rellenar nulos
df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Codificar variables categóricas con OneHotEncoding
df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

# Separar features y target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Normalizar datos numéricos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. División en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 4. Entrenamiento del modelo
print("Entrenando Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Predicciones y evaluación
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n Resultados del Modelo Random Forest")
print(f"Accuracy: {accuracy:.4f}")
print("\nMatriz de confusión:\n", cm)
print("\nReporte de clasificación:\n", report)

# 6. Graficar matriz de confusión y guardarla
OUT_IMG = os.path.join(THIS_DIR, "matriz_confusion.png")

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Matriz de Confusión (Accuracy: {accuracy:.2f})")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig(OUT_IMG)
plt.show()

print("\n Se ha guardado la matriz de confusión en 'matriz_confusion.png'")