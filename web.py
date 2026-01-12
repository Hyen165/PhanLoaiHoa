import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
from PIL import Image, ImageOps

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Function to preprocess image and make prediction
def predict_image(image: Image.Image):
    # Create array
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Resize
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    # Normalize
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    # Predict
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index][2:].strip()
    confidence_score = float(prediction[0][index])
    # Return result
    return class_name, confidence_score

# Flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    if request.method == "POST":
        file = request.files["file"]
        image = Image.open(file).convert("RGB")
        prediction, confidence = predict_image(image)
    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

# Main entry point
if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)
