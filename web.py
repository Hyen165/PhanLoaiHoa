import os
import base64
import io

os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
from PIL import Image, ImageOps

app = Flask(__name__)

# Giới hạn upload ~5MB (an toàn cho webcam base64)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# =========================
# Load YOUR model & labels
# =========================
model = tf.keras.models.load_model("keras_model.h5", compile=False)

with open("labels.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]


# =========================
# Predict function (GIỮ NGUYÊN LOGIC)
# =========================
def predict_image(image: Image.Image):
    size = (224, 224)

    image = ImageOps.fit(
        image,
        size,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(image).astype(np.float32)
    normalized_image_array = (image_array / 127.5) - 1

    data = np.expand_dims(normalized_image_array, axis=0)

    predictions = model.predict(data, verbose=0)
    index = np.argmax(predictions)
    confidence_score = float(predictions[0][index])

    class_name = class_names[index]
    if class_name[0].isdigit():
        class_name = class_name.split(" ", 1)[1]

    return class_name, confidence_score


# =========================
# Flask route
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":

        # -------- Upload image --------
        if "file" in request.files and request.files["file"].filename != "":
            try:
                image = Image.open(request.files["file"]).convert("RGB")
                prediction, confidence = predict_image(image)
            except Exception as e:
                print("Upload error:", e)

        # -------- Webcam base64 --------
        elif "webcam_image" in request.form and request.form["webcam_image"]:
            try:
                data_url = request.form["webcam_image"]
                _, encoded = data_url.split(",", 1)

                image_bytes = base64.b64decode(encoded)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                prediction, confidence = predict_image(image)
            except Exception as e:
                print("Webcam error:", e)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


# =========================
# Run
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
