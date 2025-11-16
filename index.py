from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "plant_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
]

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400

    file = request.files['image']
    img = Image.open(file.stream).convert("RGB")
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    idx = np.argmax(preds)

    return jsonify({
        "class": CLASS_NAMES[idx],
        "confidence": float(preds[idx])
    })


# ======= QUAN TRỌNG CHO RENDER =======
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
