from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.nasnet import preprocess_input
from PIL import Image

app = Flask(__name__)

# Load NASNetMobile model
MODEL_PATH = r'D:\WORK\nasnetmobile_plant_model.h5'
model = load_model(MODEL_PATH)

# Class names
class_names = ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold',
               'Septoria Leaf Spot', 'Spider Mites', 'Target Spot',
               'Yellow Curl Virus', 'Mosaic Virus', 'Healthy']

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'tomato_image' not in request.files:
            return "No file uploaded", 400

        file = request.files['tomato_image']
        if file.filename == '':
            return "No file selected", 400

        # Save uploaded image
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)

        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict using model
        preds = model.predict(img_array)
        pred_index = np.argmax(preds[0])
        prediction = class_names[pred_index]
        confidence = float(np.max(preds[0])) * 100

        # Disease info: pesticide and severity
        disease_info = {
            "Bacterial Spot": {
                "pesticide": "Copper-based bactericide",
                "severity": "High"
            },
            "Early Blight": {
                "pesticide": "Chlorothalonil or Mancozeb",
                "severity": "Medium"
            },
            "Late Blight": {
                "pesticide": "Metalaxyl or Ridomil Gold",
                "severity": "Very High"
            },
            "Leaf Mold": {
                "pesticide": "Chlorothalonil or Copper fungicides",
                "severity": "Medium"
            },
            "Septoria Leaf Spot": {
                "pesticide": "Mancozeb-based fungicide",
                "severity": "Medium"
            },
            "Spider Mites": {
                "pesticide": "Miticide or Neem oil",
                "severity": "Low"
            },
            "Target Spot": {
                "pesticide": "Strobilurin fungicides",
                "severity": "Medium"
            },
            "Yellow Curl Virus": {
                "pesticide": "No pesticide – remove infected plants, control whiteflies",
                "severity": "Very High"
            },
            "Mosaic Virus": {
                "pesticide": "No chemical – use resistant varieties and crop rotation",
                "severity": "High"
            },
            "Healthy": {
                "pesticide": "None needed",
                "severity": "None"
            }
        }

        # Fetch pesticide & severity
        info = disease_info.get(prediction, {"pesticide": "Unknown", "severity": "Unknown"})

        return render_template(
            'results.html',
            prediction=prediction,
            confidence=round(confidence, 2),
            pesticide=info["pesticide"],
            severity=info["severity"]
        )

    except Exception as e:
        return f"Error: {e}", 500

# Start the Flask server
if __name__ == '__main__':
    app.run(debug=True)
