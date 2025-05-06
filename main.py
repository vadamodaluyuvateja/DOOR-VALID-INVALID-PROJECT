from flask import Flask, render_template, request, jsonify
import os
import tempfile
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import uuid
import time
import logging
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Google Cloud Storage setup
BUCKET_NAME = "teja1234-storage-bucket"
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

# Function to ensure model is available
def ensure_model_available():
    model_path = 'yolov8n_cls_door.pt'
    
    # If model exists locally, use it directly
    if os.path.exists(model_path):
        logger.info(f"Found model locally at {model_path}")
        return model_path
    
    # If model doesn't exist locally, check if it's in bucket
    model_blob = bucket.blob('models/yolov8n_cls_door.pt')
    if model_blob.exists():
        logger.info("Downloading model from Cloud Storage")
        model_blob.download_to_filename(model_path)
        return model_path
    
    # If model isn't in bucket either, use the one we uploaded with deployment
    logger.info("Using model from deployment package")
    return model_path

# Load the YOLOv8 model
try:
    model_path = ensure_model_available()
    model = YOLO(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()
        result = {'success': False}
        
        if not model:
            logger.error("Model not loaded")
            result['error'] = 'Model not loaded properly'
            return jsonify(result)
        
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                try:
                    # Create a temp file with the correct extension
                    original_filename = secure_filename(file.filename)
                    extension = original_filename.rsplit('.', 1)[1].lower()
                    
                    # Save with proper extension
                    temp_file = tempfile.NamedTemporaryFile(suffix=f'.{extension}', delete=False)
                    temp_path = temp_file.name
                    temp_file.close()  # Close the file before writing to it
                    
                    # Save the file data to the temp path
                    file.save(temp_path)
                    
                    # Log the file details for debugging
                    logger.info(f"Saved temp file to {temp_path}, size: {os.path.getsize(temp_path)} bytes")
                    
                    # Generate unique filename for storage
                    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
                    
                    # Make prediction using the temp file
                    prediction = predict_image(temp_path)
                    
                    # Upload to Cloud Storage
                    blob = bucket.blob(f"uploads/{filename}")
                    blob.upload_from_filename(temp_path)
                    
                    # Generate public URL
                    public_url = f"https://storage.googleapis.com/{BUCKET_NAME}/uploads/{filename}"
                    
                    # Add results
                    result = {
                        'success': True,
                        'file_path': public_url,
                        'prediction': prediction['class_name'],
                        'confidence': f"{prediction['confidence']:.2%}",
                        'processing_time': f"{time.time() - start_time:.2f} seconds"
                    }
                    
                    # Clean up the temp file
                    os.unlink(temp_path)
                    
                except Exception as e:
                    logger.error(f"Error processing file: {e}")
                    result['error'] = f'Error processing file: {str(e)}'
        
        elif 'image_path' in request.form:
            # For cloud deployment, this approach might not work as expected
            result['error'] = 'Direct file path processing is not supported in cloud deployment'
        
        else:
            result['error'] = 'No file uploaded or path provided'
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        result = {'success': False, 'error': f'Unexpected error: {str(e)}'}
    
    return jsonify(result)

def predict_image(image_path):
    """Process image and return prediction results"""
    try:
        # Make prediction
        results = model.predict(source=image_path)
        
        # Get prediction details
        pred_idx = results[0].probs.top1
        class_name = model.names[pred_idx]
        confidence = results[0].probs.top1conf.item()
        
        return {
            'class_name': class_name,
            'confidence': confidence
        }
    except Exception as e:
        logger.error(f"Error predicting image: {e}")
        return {
            'class_name': 'Error',
            'confidence': 0.0
        }

# Add a health check endpoint
@app.route('/_ah/health')
def health_check():
    return 'OK', 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)