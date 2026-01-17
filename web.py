import os
import base64
import io

os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps

# =========================
# Flask app
# =========================
app = Flask(__name__)

# =========================
# Load model & labels
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "keras_model.h5")
LABEL_PATH = os.path.join(BASE_DIR, "labels.txt")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]


# =========================
# Predict function
# =========================
def predict_image(image: Image.Image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image).astype(np.float32)
    normalized_image_array = (image_array / 127.5) - 1

    data = np.expand_dims(normalized_image_array, axis=0)

    predictions = model.predict(data)
    index = int(np.argmax(predictions))
    confidence_score = float(predictions[0][index])

    class_name = class_names[index]
    if class_name and class_name[0].isdigit():
        class_name = class_name.split(" ", 1)[1]

    return class_name, confidence_score


# =========================
# Page route
# =========================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# =========================
# API route (frontend gọi)
# =========================
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.json.get("image")
        if not data:
            return jsonify(success=False, error="No image data")

        header, encoded = data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        label, confidence = predict_image(image)

        return jsonify(
            success=True,
            label=label,
            confidence=round(confidence * 100, 2)
        )

    except Exception as e:
        return jsonify(success=False, error=str(e))


# =========================
# Render entry point
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
