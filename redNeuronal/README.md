# Punto 2 – Red Neuronal

**Integrante responsable:** *Alberto Daniel Cervantes Forero*  
**Dataset:** Titanic - Machine Learning from Disaster (Kaggle)  
**Problema:** Clasificación binaria (sobrevivió / no sobrevivió)

---

## Preprocesamiento
- Se eliminaron columnas que no aportan al modelo (`PassengerId`, `Name`, `Ticket`, `Cabin`)
- Los valores nulos de **Age** se llenaron con la media y los de **Embarked** con la moda 
- Se convirtieron las variables categóricas a numéricas (OneHotEncoding)
- Se normalizaron las variables numéricas y se dividió el dataset en entrenamiento (80%) y prueba (20%)

---

## Modelo
Se implementó una red neuronal sencilla en **Keras** con dos capas ocultas y dropout para evitar sobreajuste
- Función de activación: ReLU en las capas ocultas y Sigmoid en la salida
- Optimizador: Adam
- Métrica principal: Accuracy
- Se utilizó EarlyStopping para cortar el entrenamiento cuando no había mejora

---

## Resultados
El modelo alcanzó un accuracy cercano al 0.78–0.80 en el conjunto de prueba
Se generaron los siguientes resultados:  
- Matriz de confusión
- Reporte de clasificación (precision, recall, f1-score) 
- Gráficas de la pérdida y el accuracy durante el entrenamiento

Los archivos se guardan en la carpeta `outputs_p2`

---

## Conclusiones
- En este tipo de datos tabulares pequeños, los modelos clásicos suelen ser muy fuertes 
- La red neuronal puede mejorar con más ajustes y nuevas variables, pero ya demuestra un rendimiento aceptable