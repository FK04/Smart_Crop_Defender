import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, url_for
from tensorflow.keras.preprocessing import image

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load your trained Keras model
MODEL_PATH = "Plant_disease_model_3.keras"
model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        print(f"Received message from {name} ({email}): {message}")
    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Check file presence
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400
        
        # Save the file to a temp directory inside static/
        temp_dir = os.path.join(app.static_folder, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)
        
        # Preprocess the image for prediction (*** important: normalize ***)
        try:
            img = image.load_img(file_path, target_size=(224, 224))
        except Exception as e:
            return "Error processing image", 500
        
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize by 255.0
        
        # Run model prediction
        pred = model.predict(img_array)
        
        # Use np.argmax for multi-class
        prediction_index = np.argmax(pred, axis=1)[0]
        
        # Class labels for 3 classes
        class_labels = {
            0: "Healthy",
            1: "Diseased<br>Bacterial Disease",
            2: "Diseased<br>Manganese Toxicity"
        }
        prediction_text = class_labels.get(prediction_index, "Unknown")
        
        # Build the URL for the uploaded image
        img_url = url_for('static', filename=f"temp/{file.filename}")
        return render_template('prediction.html', 
                               prediction=prediction_text, 
                               img_path=img_url)
    
    # If GET request, just show the form (prediction.html)
    return render_template('prediction.html')

if __name__ == "__main__":
    app.run(debug=True)
