import os

import numpy as np
from PIL import Image

# Silenciar logs de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gradio as gr
import keras

# 1. Cargar el modelo
MODEL_PATH = "mobilenetv2.keras"
model = keras.models.load_model(MODEL_PATH)

# 2. Cargar las etiquetas legibles desde labels.txt de forma dinámica
LABELS_PATH = "labels.txt"
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        # .strip() elimina los saltos de línea (\n) invisibles
        LABELS = [line.strip() for line in f.readlines() if line.strip()]
else:
    # Fallback de seguridad en caso de que el archivo no se encuentre
    LABELS = [f"Clase {i}" for i in range(20)]


def predict_fruit(img):
    if img is None:
        return {"error": "No image provided"}

    # Preprocesamiento para MobileNetV2
    img = img.resize((224, 224))
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Inferencia optimizada
    predictions = model.predict(img_array, verbose=0)[0]

    # Mapear los scores con las etiquetas amigables de tu archivo txt
    return {LABELS[i]: float(predictions[i]) for i in range(len(LABELS))}


# 3. Interfaz de Gradio ajustada al público final
demo = gr.Interface(
    fn=predict_fruit,
    inputs=gr.Image(type="pil", label="Captura o sube una foto"),
    outputs=gr.Label(num_top_classes=3, label="Estado del alimento"),
    title="Asistente de Clasificación de Alimentos",
    description="Identifica frutas y verduras frescas o en estado de descomposición.",
)

if __name__ == "__main__":
    demo.queue().launch()
