import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import class_weight
import joblib
import tensorflow as tf
from tensorflow import keras

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "train.csv")
OUT_DIR = os.path.join(THIS_DIR, "outputs_p2")
os.makedirs(OUT_DIR, exist_ok=True)

#Cargamos el DataSet
df = pd.read_csv(DATA_PATH)
print("Dataset cargado:", df.shape)
print(df.head(), "\n")

#Preprocesamos los datos
print("Preprocesando datos...")

#Eliminamos las columnas que no aportan al modelo
df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

#Completamos los valores faltantes
df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

#Separamos variables de entrada y la variable objetivo
y = df["Survived"].astype(int)
X = df.drop("Survived", axis=1)

#Escalamos datos numéricos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))

#Dividimos el entrenamiento y la prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

#Definimos el modelo de red neuronal
input_dim = X_train.shape[1]

#Calculamos los pesos por clase (en caso de desbalance)
cw = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {int(i): float(w) for i, w in enumerate(cw)}

#Definimos la arquitectura
model = keras.Sequential([
    keras.layers.Input(shape=(input_dim,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.20),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dropout(0.10),
    keras.layers.Dense(1, activation="sigmoid")
])

#Compilamos el modelo
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

#Entrenamos el modelo
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=1
)

#Guardamos el modelo entrenado
model_path = os.path.join(OUT_DIR, "model_nn.keras")
model.save(model_path)
print("Modelo guardado en:", model_path)

#Evaluamos el modelo
y_proba = model.predict(X_test).ravel()
y_pred = (y_proba >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=2)

print("\nResultados Red Neuronal (Keras)")
print(f"Accuracy: {acc:.4f}")
print("\nMatriz de confusión:\n", cm)
print("\nReporte de clasificación:\n", report)

#Guardamos el reporte en un archivo de texto
with open(os.path.join(OUT_DIR, "reporte_clasificacion.txt"), "w", encoding="utf-8") as f:
    f.write("Resultados Red Neuronal (Keras)\n")
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write("Matriz de confusión:\n")
    f.write(str(cm) + "\n\n")
    f.write("Reporte de clasificación:\n")
    f.write(report)

#Gráficos
plt.figure()
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Curva de Pérdida")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "curva_perdida.png"), dpi=150)

plt.figure()
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.title("Curva de Accuracy")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "curva_accuracy.png"), dpi=150)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Matriz de Confusión (Accuracy: {acc:.2f})")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "matriz_confusion_RN.png"), dpi=150)

print("\nGráficas guardadas en:", OUT_DIR)