import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import catboost as cb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib
import warnings
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download
import requests
warnings.filterwarnings('ignore')

class StrokePredictionInterface:
    def __init__(self, 
                 hf_repo_id=None, 
                 unet_model_path=None, 
                 ensemble_models_path=None,
                 hf_token=None):
        """
        Initialize the prediction interface with models from Hugging Face or local
        
        Args:
            hf_repo_id: Hugging Face repository ID (e.g., "username/stroke-detection")
            unet_model_path: Local path to U-Net model (fallback if not using HF)
            ensemble_models_path: Local path base for ensemble models (fallback)
            hf_token: Hugging Face token (optional, for private repos)
        """
        print("Loading Brain Stroke Prediction Models...")
        print("="*50)
        
        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token
        
        # Load U-Net model
        if hf_repo_id:
            print(f"Loading U-Net model from Hugging Face: {hf_repo_id}")
            self.unet_model = self._load_unet_from_hf()
        elif unet_model_path:
            print(f"Loading U-Net model from local path: {unet_model_path}")
            self.unet_model = keras.models.load_model(unet_model_path)
        else:
            raise ValueError("Either hf_repo_id or unet_model_path must be provided")
        
        # Get input specifications from U-Net
        unet_input_shape = self.unet_model.input_shape
        self.img_size = (unet_input_shape[1], unet_input_shape[2])
        self.input_channels = unet_input_shape[3] if len(unet_input_shape) > 3 else 1
        
        print(f"U-Net input shape: {unet_input_shape}")
        print(f"Image size: {self.img_size}")
        print(f"Input channels: {self.input_channels}")
        
        # Load ensemble models
        print("\nLoading ensemble models...")
        if hf_repo_id:
            self._load_ensemble_from_hf()
        else:
            self._load_ensemble_local(ensemble_models_path)
        
        # Check if all models are loaded
        self.models_loaded = all([
            self.lightgbm_model is not None,
            self.catboost_model is not None,
            self.adaboost_model is not None,
            self.decision_tree_meta is not None
        ])
        
        if self.models_loaded:
            print("\n✓ All models loaded successfully!")
        else:
            print("\n⚠ Some models failed to load. Predictions may not work properly.")
    
    def _load_unet_from_hf(self):
        """Load U-Net model from Hugging Face"""
        try:
            # Download model file from Hugging Face
            model_path = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename="best_unet_model.h5",
                token=self.hf_token,
                cache_dir="./model_cache"
            )
            print(f"✓ Downloaded U-Net model to: {model_path}")
            
            # Load the model
            model = keras.models.load_model(model_path)
            return model
        except Exception as e:
            print(f"✗ Error loading U-Net from Hugging Face: {e}")
            raise
    
    def _load_ensemble_from_hf(self):
        """Load ensemble models from Hugging Face"""
        model_files = {
            'lightgbm': 'my_stroke_ensemble_lightgbm.pkl',
            'catboost': 'my_stroke_ensemble_catboost.pkl',
            'adaboost': 'my_stroke_ensemble_adaboost.pkl',
            'decision_tree_meta': 'my_stroke_ensemble_decision_tree_meta.pkl',
            'scaler': 'my_stroke_ensemble_scaler.pkl'
        }
        
        for model_name, filename in model_files.items():
            try:
                # Download model file
                model_path = hf_hub_download(
                    repo_id=self.hf_repo_id,
                    filename=filename,
                    token=self.hf_token,
                    cache_dir="./model_cache"
                )
                
                # Load the model
                model = joblib.load(model_path)
                setattr(self, f"{model_name}_model" if model_name != 'scaler' else model_name, model)
                print(f"✓ {model_name.replace('_', ' ').title()} loaded")
                
            except Exception as e:
                print(f"✗ Error loading {model_name}: {e}")
                setattr(self, f"{model_name}_model" if model_name != 'scaler' else model_name, None)
        
        # Fallback scaler if not loaded
        if not hasattr(self, 'scaler') or self.scaler is None:
            self.scaler = StandardScaler()
    
    def _load_ensemble_local(self, ensemble_models_path):
        """Load ensemble models from local files"""
        try:
            self.lightgbm_model = joblib.load(f"{ensemble_models_path}_lightgbm.pkl")
            print("✓ LightGBM model loaded")
        except:
            print("✗ Error loading LightGBM model")
            self.lightgbm_model = None
            
        try:
            self.catboost_model = joblib.load(f"{ensemble_models_path}_catboost.pkl")
            print("✓ CatBoost model loaded")
        except:
            print("✗ Error loading CatBoost model")
            self.catboost_model = None
            
        try:
            self.adaboost_model = joblib.load(f"{ensemble_models_path}_adaboost.pkl")
            print("✓ AdaBoost model loaded")
        except:
            print("✗ Error loading AdaBoost model")
            self.adaboost_model = None
            
        try:
            self.decision_tree_meta = joblib.load(f"{ensemble_models_path}_decision_tree_meta.pkl")
            print("✓ Decision Tree meta-classifier loaded")
        except:
            print("✗ Error loading Decision Tree meta-classifier")
            self.decision_tree_meta = None
            
        try:
            self.scaler = joblib.load(f"{ensemble_models_path}_scaler.pkl")
            print("✓ Feature scaler loaded")
        except:
            print("✗ Error loading feature scaler")
            self.scaler = StandardScaler()
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for prediction
        
        Args:
            image_path: Path to the CT scan image
            
        Returns:
            Preprocessed image array
        """
        try:
            color_mode = 'grayscale' if self.input_channels == 1 else 'rgb'
            img = load_img(image_path, target_size=self.img_size, color_mode=color_mode)
            img_array = img_to_array(img) / 255.0
            
            # Ensure correct channels
            if self.input_channels == 1 and len(img_array.shape) == 3 and img_array.shape[-1] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                img_array = np.expand_dims(img_array, axis=-1)
            elif self.input_channels == 3 and len(img_array.shape) == 3 and img_array.shape[-1] == 1:
                img_array = np.repeat(img_array, 3, axis=-1)
            
            return img_array
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    
    def extract_unet_features(self, img_array):
        """
        Extract features from preprocessed image using U-Net
        
        Args:
            img_array: Preprocessed image array
            
        Returns:
            Extracted features
        """
        try:
            # Add batch dimension
            img_batch = np.expand_dims(img_array, axis=0)
            
            # Find suitable layer for feature extraction
            layer_info = []
            for i, layer in enumerate(self.unet_model.layers):
                try:
                    output_shape = layer.output_shape
                    if isinstance(output_shape, tuple) and len(output_shape) == 4:
                        h, w = output_shape[1], output_shape[2]
                        if h is not None and w is not None and h <= 56 and w <= 56:
                            layer_info.append((i, layer.name, output_shape))
                except:
                    continue
            
            # Use a suitable layer or fallback
            if layer_info:
                feature_layer_name = layer_info[0][1]
            else:
                mid_idx = len(self.unet_model.layers) // 2
                feature_layer_name = self.unet_model.layers[mid_idx].name
            
            # Create feature extractor with global average pooling
            base_output = self.unet_model.get_layer(feature_layer_name).output
            
            if len(base_output.shape) == 4:
                pooled_output = tf.keras.layers.GlobalAveragePooling2D()(base_output)
            else:
                pooled_output = base_output
            
            feature_extractor = keras.Model(
                inputs=self.unet_model.input,
                outputs=pooled_output
            )
            
            # Extract features
            features = feature_extractor.predict(img_batch, verbose=0)
            
            return features.flatten()
            
        except Exception as e:
            print(f"Error extracting U-Net features: {str(e)}")
            return self.create_simple_features(img_array)
    
    def create_simple_features(self, img_array):
        """
        Create simple statistical features from image
        
        Args:
            img_array: Image array
            
        Returns:
            Statistical features
        """
        # Convert to grayscale for analysis
        if len(img_array.shape) == 3:
            if img_array.shape[-1] == 3:
                gray_img = cv2.cvtColor(img_array.astype(np.float32), cv2.COLOR_RGB2GRAY)
            else:
                gray_img = img_array[:, :, 0].astype(np.float32)
        else:
            gray_img = img_array.astype(np.float32)
        
        # Calculate statistics
        features = [
            np.mean(gray_img),
            np.std(gray_img),
            np.min(gray_img),
            np.max(gray_img),
            np.median(gray_img),
            np.percentile(gray_img, 25),
            np.percentile(gray_img, 75)
        ]
        
        # Add histogram features
        hist = cv2.calcHist([gray_img], [0], None, [16], [gray_img.min(), gray_img.max()])
        hist_features = hist.flatten() / (hist.sum() + 1e-7)
        
        return np.array(features + hist_features.tolist())
    
    def predict_single_image(self, image_path):
        """
        Predict stroke probability for a single image
        
        Args:
            image_path: Path to CT scan image
            
        Returns:
            Dictionary containing prediction results
        """
        if not self.models_loaded:
            return {"error": "Models not properly loaded"}
        
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        if img_array is None:
            return {"error": "Failed to preprocess image"}
        
        try:
            # Extract features
            unet_features = self.extract_unet_features(img_array)
            tabular_features = self.create_simple_features(img_array)
            
            # Get expected feature counts
            lgb_features = getattr(self.lightgbm_model, 'n_features_in_', None) or getattr(self.lightgbm_model, 'n_features_', None)
            
            # Create features to match model expectations
            if lgb_features is not None:
                target_features = lgb_features
                combined_raw = np.hstack([unet_features, tabular_features])
                
                if len(combined_raw) == target_features:
                    model_features = combined_raw.reshape(1, -1)
                elif len(combined_raw) > target_features:
                    model_features = combined_raw[:target_features].reshape(1, -1)
                else:
                    padded = np.zeros(target_features)
                    padded[:len(combined_raw)] = combined_raw
                    model_features = padded.reshape(1, -1)
                
                # Scale features
                try:
                    model_features = self.scaler.transform(model_features)
                except:
                    pass  # Use unscaled if scaling fails
            else:
                model_features = np.hstack([unet_features, tabular_features]).reshape(1, -1)
            
            # Get predictions from individual models
            lgb_pred = self.lightgbm_model.predict_proba(model_features)[0, 1]
            cat_pred = self.catboost_model.predict_proba(model_features)[0, 1]
            ada_pred = self.adaboost_model.predict_proba(model_features)[0, 1]
            
            # Meta-features for decision tree
            meta_features = np.array([[lgb_pred, cat_pred, ada_pred]])
            
            # Final prediction
            final_prediction = self.decision_tree_meta.predict(meta_features)[0]
            final_probabilities = self.decision_tree_meta.predict_proba(meta_features)[0]
            
            # Prepare results
            result = {
                "image_path": image_path,
                "prediction": int(final_prediction),
                "prediction_label": "Stroke Detected" if final_prediction == 1 else "Normal",
                "stroke_probability": float((lgb_pred + cat_pred + ada_pred) / 3),
                "individual_predictions": {
                    "lightgbm": float(lgb_pred),
                    "catboost": float(cat_pred), 
                    "adaboost": float(ada_pred)
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            return result
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Full error details:\n{error_details}")
            return {"error": f"Prediction failed: {str(e)}", "details": error_details}
    
    def predict_multiple_images(self, image_paths):
        """
        Predict on multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        print("-" * 50)
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"Processing image {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            result = self.predict_single_image(image_path)
            results.append(result)
            
            if "error" not in result:
                prediction = result["prediction_label"]
                print(f"Result: {prediction}")
            else:
                print(f"Error: {result['error']}")
            
            print()
        
        return results


# Example usage
def main():
    """
    Main function demonstrating how to use the prediction interface with Hugging Face
    """
    # Initialize with Hugging Face repo
    predictor = StrokePredictionInterface(
        hf_repo_id="mv2274/CNNBasedBrainStrokeDetection",  # Replace with your HF repo
        hf_token=os.getenv('HF_TOKEN', '')  # Add token if private repo
    )
    
    # Example: Predict on a single image
    print("\n" + "="*50)
    print("SINGLE IMAGE PREDICTION")
    print("="*50)
    
    single_image_path = "test_image.jpg"
    if os.path.exists(single_image_path):
        result = predictor.predict_single_image(single_image_path)
        
        if "error" not in result:
            print(f"Image: {os.path.basename(result['image_path'])}")
            print(f"Prediction: {result['prediction_label']}")
            print(f"Stroke Probability: {result['stroke_probability']:.3f}")
        else:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()