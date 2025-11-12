# 🧠 Tarea Semana 3 — Clasificación de Imágenes y Visualización de Características con CNN

🧑‍💻 Autor

Oscar Albuja
Estudiante Maestría Inteligencia Artificial Aplicada — Universidad de las Américas (UDLA)
Proyecto desarrollado como parte de la Semana 3 - Redes Neuronales Profundas

## 📝 Descripción general
Esta actividad tiene como objetivo **implementar una red neuronal profunda (CNN o LSTM)** para resolver un problema de **clasificación supervisada**, explorando arquitecturas más avanzadas que las redes MLP.  
El enfoque está en **patrones secuenciales** (LSTM) o **patrones espaciales/locales** (CNN), entendiendo su funcionamiento básico y cómo se entrenan en la práctica.

En este caso, se desarrolló una **Red Neuronal Convolucional (CNN)** aplicada al dataset **MNIST**, que contiene imágenes de dígitos escritos a mano (0–9).

---

## 🎯 Objetivo
Aplicar un modelo de red neuronal profunda especializado (**CNN o LSTM**) para aprender a **clasificar imágenes o secuencias** a partir de ejemplos previamente etiquetados, evaluando su rendimiento con métricas cuantitativas y visualizaciones de activación.

---

## 📊 Tipo de problema
**Clasificación supervisada**

El modelo aprende a asignar etiquetas de clase en función de los patrones detectados en los datos.

- **Binaria:** dos clases (ej. positivo / negativo).  
- **Multiclase:** más de dos clases (ej. categorías 0–9 en MNIST).

---

## 🔍 Variable objetivo
- Variable **categórica**.  
- El modelo predice la clase correcta (dígito 0–9) según los **patrones espaciales** detectados en la imagen.

---

## ⚙️ Arquitectura del modelo CNN

1. **Capa de entrada**  
   - Tamaño `(28, 28, 1)` correspondiente a las imágenes en escala de grises del dataset MNIST.

2. **Capas convolucionales y de pooling**  
   - Dos bloques `Conv2D` con 8 y 16 filtros (kernel 3×3) y activación ReLU.  
   - Cada bloque seguido por `MaxPooling2D (2×2)` para reducir la dimensionalidad.  
   - Estas capas detectan **bordes, contornos, curvas y estructuras locales** de los dígitos.

3. **Capas densas (fully connected)**  
   - `Flatten` → `Dense(64, relu)` → `Dense(10, softmax)`  
   - La última capa con 10 neuronas representa las 10 clases posibles.

4. **Compilación y entrenamiento**
   ```python
   model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
   model.fit(x_train, y_train, epochs=5, validation_split=0.1)
