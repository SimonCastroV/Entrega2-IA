# Entrega2-IA/GBC/GradientBC.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ruta robusta al CSV, relativa a este archivo
BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.normpath(os.path.join(BASE_DIR, "../data/train.csv"))
OUT_DIR = os.path.join(BASE_DIR, "outputs_gbc")
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Cargar datos
df = pd.read_csv(CSV_PATH)

# 2) Variables
X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"])
y = df["Survived"]
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns

# 3) Preprocesamiento
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_features),
    ("cat", categorical_transformer, cat_features)
])

# 4) Modelo
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

clf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", gb_model)
])

# 5) Train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 6) Entrenar y evaluar
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print("\nClassification report:\n")
print(report)
print("\nMatriz de confusión:\n")
print(cm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Matriz de Confusión Gradient Boosting (Accuracy: {acc:.2f})")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "matriz_confusion_gbc.png"), dpi=150)
plt.close()

print("\nMatriz de confusión guardada en:", os.path.join(OUT_DIR, "matriz_confusion_gbc.png"))
