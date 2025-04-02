import tensorflow as tf
from PIL import Image
import numpy as np
import os
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import io
import shutil

app = Flask(__name__)
CORS(app)  # Allow CORS for all domains

# Load the trained model
model_path = 'finetuned_mobilenetv2.h5'
model = tf.keras.models.load_model(model_path)
img_height, img_width = 128, 128
feedback_folder = 'feedback'
batch_size = 10  # Set the batch size threshold for retraining

# Ensure feedback folder exists
os.makedirs(feedback_folder, exist_ok=True)

# Preprocess the image
def process_image(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    edges = cv2.Canny(image, 50, 150)
    edges_colored = cv2.merge([edges, edges, edges])
    combined_image = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)
    combined_image = cv2.resize(combined_image, (img_width, img_height))
    combined_image = combined_image / 255.0
    combined_image = np.expand_dims(combined_image, axis=0)
    return combined_image

# Store feedback for later batch training
def store_feedback(image, true_label):
    label_dir = os.path.join(feedback_folder, str(true_label))
    os.makedirs(label_dir, exist_ok=True)
    
    # Save the image as PNG
    image_count = len(os.listdir(label_dir))
    image_filename = f"feedback_{image_count + 1}.png"
    image_path = os.path.join(label_dir, image_filename)
    image.save(image_path)

# Batch training logic
def batch_retrain():
    all_images = []
    all_labels = []

    for label in os.listdir(feedback_folder):
        label_dir = os.path.join(feedback_folder, label)
        for image_file in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_file)
            image = Image.open(image_path)
            processed_image = process_image(image)
            all_images.append(processed_image)
            all_labels.append(int(label))
    
    if len(all_images) >= batch_size:
        # Convert to array for training
        all_images = np.concatenate(all_images, axis=0)
        all_labels = tf.keras.utils.to_categorical(all_labels, num_classes=4)
        
        # Train the model with feedback
        model.fit(all_images, all_labels, epochs=1, batch_size=batch_size)
        
        # Save the model
        temp_model_path = model_path.replace('.h5', '_temp.h5')
        model.save(temp_model_path)
        os.rename(temp_model_path, model_path)
        
        # Clear the feedback folder after training
        shutil.rmtree(feedback_folder)
        os.makedirs(feedback_folder, exist_ok=True)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file:
        try:
            image = Image.open(file)
            img_array = process_image(image)
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            class_labels = ['cup', 'knife', 'scissors', 'spoon']
            predicted_label = class_labels[predicted_class_index] if class_labels else predicted_class_index
            return jsonify({'predicted_class': predicted_label})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'File not found'}), 400

# Feedback route
@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.form
    predicted_class = data['predicted_class']
    true_class = data['true_class']
    image = data['image']  # Base64 image

    class_labels = ['cup', 'knife', 'scissors', 'spoon']  # Update with actual class labels
    true_label_index = class_labels.index(true_class)

    # Store the image in the feedback folder
    img = Image.open(io.BytesIO(base64.b64decode(image.split(',')[1])))
    store_feedback(img, true_label_index)
    
    # Check if we need to retrain
    batch_retrain()

    return jsonify({'message': 'Feedback received.'})

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)