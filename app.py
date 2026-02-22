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
print("Initializing prediction models from Hugging Face...")
print("="*60)

# Get Hugging Face repo and token from environment variables
HF_REPO_ID = os.getenv('HF_REPO_ID', 'mv2274/CNNBasedBrainStrokeDetection')
HF_TOKEN = os.getenv('HF_TOKEN', '')  # Optional, for private repos

try:
    predictor = StrokePredictionInterface(
        hf_repo_id=HF_REPO_ID,
        hf_token=HF_TOKEN
    )
    print("✅ Models loaded successfully from Hugging Face!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Trying to load from local files as fallback...")
    try:
        predictor = StrokePredictionInterface(
            unet_model_path="best_unet_model.h5",
            ensemble_models_path="my_stroke_ensemble"
        )
        print("✅ Models loaded from local files!")
    except Exception as e2:
        print(f"❌ Failed to load models: {e2}")
        predictor = None

print("="*60)

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
        'status': 'healthy' if predictor and predictor.models_loaded else 'unhealthy',
        'models_loaded': predictor.models_loaded if predictor else False,
        'model_source': 'huggingface',
        'hf_repo': HF_REPO_ID,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Accepts multiple image files and returns predictions
    """
    if not predictor or not predictor.models_loaded:
        return jsonify({
            'error': 'Models not loaded. Please check server logs.',
            'success': False
        }), 503
    
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
                original_name = file.filename
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                unique_filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                
                # Run prediction
                prediction_result = predictor.predict_single_image(filepath)
                
                prediction_result['original_filename'] = original_name
                prediction_result['saved_filename'] = unique_filename
                prediction_result['image_path'] = filepath
                
                # Add base64 encoded image for display
                if "error" not in prediction_result:
                    image_base64 = encode_image_to_base64(filepath)
                    if image_base64:
                        prediction_result['image_data'] = f"data:image/jpeg;base64,{image_base64}"
                    
                    # Optionally delete file after prediction to save space
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
            'total_processed': len(results),
            'model_source': 'huggingface'
        })
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in prediction: {error_trace}")
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
    if not predictor or not predictor.models_loaded:
        return jsonify({
            'error': 'Models not loaded',
            'success': False
        }), 503
    
    try:
        files = request.files.getlist('files')
        
        if not files:
            return jsonify({'error': 'No files provided'}), 400
        
        # Save all files first
        image_paths = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
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
            'total_processed': len(results),
            'model_source': 'huggingface'
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}',
            'success': False
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """
    Get information about loaded models
    """
    if not predictor:
        return jsonify({'error': 'Models not initialized'}), 503
    
    return jsonify({
        'models_loaded': predictor.models_loaded,
        'model_source': 'huggingface',
        'hf_repo': HF_REPO_ID,
        'image_size': predictor.img_size,
        'input_channels': predictor.input_channels,
        'models': {
            'unet': predictor.unet_model is not None,
            'lightgbm': predictor.lightgbm_model is not None,
            'catboost': predictor.catboost_model is not None,
            'adaboost': predictor.adaboost_model is not None,
            'decision_tree_meta': predictor.decision_tree_meta is not None
        }
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧠 Brain Stroke Detection Server Starting...")
    print("="*60)
    if predictor:
        print(f"✅ Models loaded: {predictor.models_loaded}")
        print(f"📦 Model source: Hugging Face ({HF_REPO_ID})")
    else:
        print("❌ Models failed to load")
    print("🌐 Server will be available at: http://localhost:5000")
    print("="*60 + "\n")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)