from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = tf.keras.models.load_model('acne_model.h5')

# Class labels corresponding to your dataset folder names
class_labels = ['Acne', 'Clear', 'Comedo']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to preprocess the image
def preprocess_image(img):
    img = img.resize((150, 150))  # Resize to match the input size of the model
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for uploading and predicting
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Open and process the image for prediction
        with Image.open(filepath) as img:
            img_array = preprocess_image(img)
            
            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            class_name = class_labels[predicted_class]
            
            # Calculate confidence score
            confidence_score = float(prediction[0][predicted_class] * 100)
            
            # Generate relative path for template
            relative_path = os.path.join('uploads', filename)
            
            # Return the result page with the prediction and image path
            return render_template('result.html', 
                                class_name=class_name,
                                confidence_score=confidence_score,
                                image_path=relative_path)

# Add route to serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)