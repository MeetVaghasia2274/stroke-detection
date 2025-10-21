from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import base64
from werkzeug.utils import secure_filename
import io
from PIL import Image
import numpy as np
from datetime import datetime

# Import your prediction interface
from stroke_prediction import StrokePredictionInterface

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for frontend

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize predictor (load models once at startup)
print("Initializing prediction models...")
predictor = StrokePredictionInterface(
    unet_model_path="best_unet_model.h5",
    ensemble_models_path="my_stroke_ensemble"
)
print("Models loaded successfully!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_to_base64(image_path):
    """Convert image to base64 for sending to frontend"""
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except:
        return None

@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('static', 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': predictor.models_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Accepts multiple image files and returns predictions
    """
    try:
        # Check if files were uploaded
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        results = []
        
        for file in files:
            if file and allowed_file(file.filename):
                # Save file securely
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                unique_filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                
                # Run prediction
                prediction_result = predictor.predict_single_image(filepath)
                
                # Add base64 encoded image for display
                if "error" not in prediction_result:
                    image_base64 = encode_image_to_base64(filepath)
                    if image_base64:
                        prediction_result['image_data'] = f"data:image/jpeg;base64,{image_base64}"
                    
                    # Clean up: optionally delete file after prediction
                    # os.remove(filepath)
                
                results.append(prediction_result)
            else:
                results.append({
                    'error': f'Invalid file type: {file.filename}',
                    'filename': file.filename
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results)
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}',
            'success': False
        }), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint for multiple images
    More efficient for large batches
    """
    try:
        files = request.files.getlist('files')
        
        if not files:
            return jsonify({'error': 'No files provided'}), 400
        
        # Save all files first
        image_paths = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                unique_filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                image_paths.append(filepath)
        
        # Batch prediction
        results = predictor.predict_multiple_images(image_paths)
        
        # Add base64 encoded images
        for result in results:
            if "error" not in result:
                image_base64 = encode_image_to_base64(result['image_path'])
                if image_base64:
                    result['image_data'] = f"data:image/jpeg;base64,{image_base64}"
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results)
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}',
            'success': False
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    """
    Get overall statistics (if you want to track predictions)
    """
    # You could implement database tracking here
    return jsonify({
        'message': 'Statistics endpoint - implement database tracking for production'
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Brain Stroke Detection Server Starting...")
    print("="*60)
    print(f"Models loaded: {predictor.models_loaded}")
    print("Server will be available at: http://localhost:5000")
    print("="*60 + "\n")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)