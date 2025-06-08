import os
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestClassifier
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Class labels
CLASS_LABELS = {
    1: 'glass',    # Changed to lowercase to match training
    2: 'paper',
    3: 'cardboard',
    4: 'plastic',
    5: 'metal',
    6: 'trash'
}

# Category information
CATEGORY_INFO = {
    'glass': 'Glass items are highly recyclable. Clean and separate from other materials.',
    'paper': 'Paper products can be recycled multiple times. Keep them dry and clean.',
    'cardboard': 'Flatten cardboard boxes and keep them dry for recycling.',
    'plastic': 'Check the recycling number and clean before recycling.',
    'metal': 'Metal items are valuable recyclables. Clean and crush if possible.',
    'trash': 'Non-recyclable items. Dispose in regular waste.'
}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(128, 128)):  # Changed to 128x128
    """Preprocess image for prediction"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise Exception(f"Could not read image: {image_path}")
            
        # Resize to fixed size
        img = cv2.resize(img, target_size)
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Additional preprocessing steps to match training
        # 1. Contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # 2. Normalize pixel values
        img_normalized = enhanced.astype('float32') / 255.0
        
        # 3. Extract features
        # Flatten image
        img_flat = img_normalized.reshape(-1)
        
        # Color features (mean RGB values)
        color_features = np.mean(img_normalized, axis=(0,1))
        
        # Texture features (std of RGB channels)
        texture_features = np.std(img_normalized, axis=(0,1))
        
        # Combine all features
        features = np.concatenate([
            img_flat,              # Flattened image
            color_features,        # Color features
            texture_features       # Texture features
        ])
        
        # Reshape for model
        features = features.reshape(1, -1)
        return features
    
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

# Load the model
model = None
try:
    with open('model/random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
except:
    print("Model file not found. Please train the model first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess the image
            processed_image = preprocess_image(filepath)
            
            if processed_image is not None and model is not None:
                # Make prediction
                prediction = model.predict(processed_image)
                # Get prediction probability
                prob = model.predict_proba(processed_image)
                max_prob = np.max(prob) * 100
                
                # Get class label
                predicted_class = CLASS_LABELS[prediction[0]]
                
                result = {
                    'prediction': predicted_class,
                    'probability': float(max_prob),
                    'image_path': os.path.join('uploads', filename),
                    'category_info': CATEGORY_INFO[predicted_class]
                }
                return jsonify(result)
            else:
                return jsonify({'error': 'Error processing image or model not loaded'})
        except Exception as e:
            return jsonify({'error': f'An error occurred during classification: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True) 