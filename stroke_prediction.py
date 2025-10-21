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
warnings.filterwarnings('ignore')

class StrokePredictionInterface:
    def __init__(self, unet_model_path, ensemble_models_path):
        """
        Initialize the prediction interface with trained models
        
        Args:
            unet_model_path: Path to the trained U-Net .h5 model
            ensemble_models_path: Base path for ensemble models (without extensions)
        """
        print("Loading Brain Stroke Prediction Models...")
        print("="*50)
        
        # Load U-Net model
        print("Loading U-Net model...")
        self.unet_model = keras.models.load_model(unet_model_path)
        
        # Get input specifications from U-Net
        unet_input_shape = self.unet_model.input_shape
        self.img_size = (unet_input_shape[1], unet_input_shape[2])
        self.input_channels = unet_input_shape[3] if len(unet_input_shape) > 3 else 1
        
        print(f"U-Net input shape: {unet_input_shape}")
        print(f"Image size: {self.img_size}")
        print(f"Input channels: {self.input_channels}")
        
        # Load ensemble models
        print("\nLoading ensemble models...")
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
            
            # Find suitable layer for feature extraction (memory-efficient)
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
                feature_layer_name = layer_info[0][1]  # Use first suitable layer
            else:
                # Fallback to middle layer
                mid_idx = len(self.unet_model.layers) // 2
                feature_layer_name = self.unet_model.layers[mid_idx].name
            
            # Create feature extractor with global average pooling
            base_output = self.unet_model.get_layer(feature_layer_name).output
            
            if len(base_output.shape) == 4:  # (batch, H, W, C)
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
            # Fallback to simple statistics
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
            
            #print(f"U-Net features shape: {unet_features.shape}")
            #print(f"Tabular features shape: {tabular_features.shape}")
            
            # Get expected feature counts from models
            scaler_features = getattr(self.scaler, 'n_features_in_', None)
            lgb_features = getattr(self.lightgbm_model, 'n_features_in_', None) or getattr(self.lightgbm_model, 'n_features_', None)
            
            #print(f"Scaler expects: {scaler_features} features")
            #print(f"LightGBM expects: {lgb_features} features")
            
            # Strategy: Create features to match what the models actually expect
            if lgb_features is not None:
                target_features = lgb_features
                #print(f"Target feature count: {target_features}")
                
                # Create combined features first
                combined_raw = np.hstack([unet_features, tabular_features])
                #print(f"Combined raw features shape: {combined_raw.shape}")
                
                if len(combined_raw) == target_features:
                    #print("Perfect match with combined features")
                    model_features = combined_raw.reshape(1, -1)
                elif len(combined_raw) > target_features:
                    #print(f"Truncating from {len(combined_raw)} to {target_features}")
                    model_features = combined_raw[:target_features].reshape(1, -1)
                else:
                    #print(f"Padding from {len(combined_raw)} to {target_features}")
                    padded = np.zeros(target_features)
                    padded[:len(combined_raw)] = combined_raw
                    model_features = padded.reshape(1, -1)
                
                # Handle scaler mismatch
                if scaler_features is not None and scaler_features != target_features:
                    #print(f"Scaler/model mismatch: scaler expects {scaler_features}, models expect {target_features}")
                    
                    # Try to create features that work with scaler
                    if scaler_features == len(tabular_features):
                        #print("Scaler was trained on tabular features only")
                        # Scale only the tabular part, leave U-Net features unscaled
                        scaled_tabular = self.scaler.transform(tabular_features.reshape(1, -1))
                        
                        # Combine scaled tabular with raw U-Net features
                        if len(unet_features) + len(scaled_tabular.flatten()) == target_features:
                            model_features = np.hstack([unet_features, scaled_tabular.flatten()]).reshape(1, -1)
                        else:
                            # Adjust U-Net features to fit
                            remaining_features = target_features - len(scaled_tabular.flatten())
                            if len(unet_features) >= remaining_features:
                                adjusted_unet = unet_features[:remaining_features]
                            else:
                                adjusted_unet = np.pad(unet_features, (0, remaining_features - len(unet_features)))
                            model_features = np.hstack([adjusted_unet, scaled_tabular.flatten()]).reshape(1, -1)
                        
                        #print(f"Created mixed features shape: {model_features.shape}")
                    else:
                        # Create a new scaler or skip scaling
                        #print("Warning: Creating features without proper scaling")
                        model_features = model_features  # Use as-is without scaling
                else:
                    # Scale normally
                    model_features = self.scaler.transform(model_features)
                    
            else:
                #print("Cannot determine expected feature count, using combined features")
                model_features = np.hstack([unet_features, tabular_features]).reshape(1, -1)
                
                # Try scaling, handle errors gracefully
                try:
                    model_features = self.scaler.transform(model_features)
                except ValueError as e:
                    print(f"Scaling failed: {e}, using unscaled features")
            
            #print(f"Final model features shape: {model_features.shape}")
            
            # Get predictions from individual models
            try:
                lgb_pred = self.lightgbm_model.predict_proba(model_features)[0, 1]
                print(f"LightGBM prediction: {lgb_pred}")
            except Exception as e:
                print(f"LightGBM error: {e}")
                lgb_pred = 0.5  # Fallback
            
            try:
                cat_pred = self.catboost_model.predict_proba(model_features)[0, 1]
                print(f"CatBoost prediction: {cat_pred}")
            except Exception as e:
                print(f"CatBoost error: {e}")
                cat_pred = 0.5  # Fallback
                
            try:
                ada_pred = self.adaboost_model.predict_proba(model_features)[0, 1]
                print(f"AdaBoost prediction: {ada_pred}")
            except Exception as e:
                print(f"AdaBoost error: {e}")
                ada_pred = 0.5  # Fallback
            
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
                "stroke_probability": float((lgb_pred+cat_pred+ada_pred)/3),
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
    
    def visualize_results(self, results, save_path=None):
        """
        Create a clean visualization with images, pie chart, and model comparison
        
        Args:
            results: List of prediction results
            save_path: Path to save the visualization
        """
        # Filter out error results and limit to 10 images
        valid_results = [r for r in results if "error" not in r][:10]
        
        if not valid_results:
            print("No valid results to visualize")
            return
        
        # Create figure with proper layout
        fig = plt.figure(figsize=(20, 12))
        
        # Define grid layout: 3 rows, 5 columns
        # Row 1-2: Images (2 rows x 5 columns = 10 images)
        # Row 3: Pie chart (left) and Model comparison (right)
        
        n_images = len(valid_results)
        
        # Display images in 2x5 grid
        for i, result in enumerate(valid_results):
            row = i // 5
            col = i % 5
            
            # Calculate subplot position (1-10 for images)
            subplot_num = row * 5 + col + 1
            ax = plt.subplot(3, 5, subplot_num)
            
            try:
                # Load and display image
                img = plt.imread(result["image_path"])
                ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                
                # Add prediction text with colored background
                prediction_text = f"{result['prediction_label']}"
                color = 'red' if result['prediction'] == 1 else 'green'
                
                ax.text(0.02, 0.98, prediction_text, transform=ax.transAxes, 
                       verticalalignment='top', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8),
                       fontsize=10, color='white', weight='bold')
                
                # Clean title
                filename = os.path.basename(result['image_path'])
                if len(filename) > 15:
                    filename = filename[:12] + "..."
                ax.set_title(f"{filename}", fontsize=9, pad=5)
                ax.axis('off')
                
            except Exception as e:
                print(f"Error loading image {result['image_path']}: {e}")
                ax.text(0.5, 0.5, 'Image\nLoad Error', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10)
                ax.axis('off')
        
        # Hide unused image slots if less than 10 images
        for i in range(n_images, 10):
            row = i // 5
            col = i % 5
            subplot_num = row * 5 + col + 1
            ax = plt.subplot(3, 5, subplot_num)
            ax.axis('off')
        
        # Summary statistics for charts
        stroke_count = sum(1 for r in valid_results if r['prediction'] == 1)
        normal_count = len(valid_results) - stroke_count
        
        # Pie chart (bottom left, spanning 2 columns)
        pie_ax = plt.subplot(3, 5, (11, 12))  # Positions 11-12
        labels = ['Normal', 'Stroke']
        sizes = [normal_count, stroke_count]
        colors = ['lightgreen', 'lightcoral']
        explode = (0.05, 0.05)
        
        wedges, texts, autotexts = pie_ax.pie(sizes, explode=explode, labels=labels, colors=colors, 
                                             autopct='%1.1f%%', shadow=True, startangle=90,
                                             textprops={'fontsize': 12, 'weight': 'bold'})
        
        # Make percentage text more visible
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(11)
            autotext.set_weight('bold')
        
        pie_ax.set_title(f'Prediction Summary\n({len(valid_results)} images)', 
                        fontsize=14, weight='bold', pad=20)
        
        # Model comparison chart (bottom right, spanning 3 columns)
        model_ax = plt.subplot(3, 5, (13, 15))  # Positions 13-15
        
        # Calculate average predictions for each model
        models = ['LightGBM', 'CatBoost', 'AdaBoost']
        model_keys = ['lightgbm', 'catboost', 'adaboost']
        avg_predictions = []
        
        for model_key in model_keys:
            avg_pred = np.mean([r['individual_predictions'][model_key] for r in valid_results])
            avg_predictions.append(avg_pred)
        
        # Create bar chart
        bars = model_ax.bar(models, avg_predictions, 
                           color=['skyblue', 'orange', 'lightgreen'],
                           edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Customize model comparison chart
        model_ax.set_ylabel('Average Stroke Probability', fontsize=12, weight='bold')
        model_ax.set_title('Individual Model Performance', fontsize=14, weight='bold', pad=20)
        model_ax.set_ylim(0, 1)
        model_ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar, pred in zip(bars, avg_predictions):
            height = bar.get_height()
            model_ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                         f'{pred:.3f}', ha='center', va='bottom', 
                         fontsize=11, weight='bold')
        
        # Style the model comparison
        model_ax.tick_params(axis='x', labelsize=11)
        model_ax.tick_params(axis='y', labelsize=10)
        
        # Add overall title
        fig.suptitle('Brain Stroke Detection Results', fontsize=18, weight='bold', y=0.98)
        
        # Adjust layout for better spacing
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def generate_report(self, results, save_path=None):
        """
        Generate a detailed text report of the predictions
        
        Args:
            results: List of prediction results
            save_path: Path to save the report
        """
        valid_results = [r for r in results if "error" not in r]
        error_results = [r for r in results if "error" in r]
        
        report = []
        report.append("BRAIN STROKE PREDICTION REPORT")
        report.append("=" * 50)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total images processed: {len(results)}")
        report.append(f"Successful predictions: {len(valid_results)}")
        report.append(f"Failed predictions: {len(error_results)}")
        report.append("")
        
        # Summary statistics
        if valid_results:
            stroke_count = sum(1 for r in valid_results if r['prediction'] == 1)
            normal_count = len(valid_results) - stroke_count
            
            report.append("SUMMARY STATISTICS")
            report.append("-" * 30)
            report.append(f"Normal scans: {normal_count}")
            report.append(f"Stroke detected: {stroke_count}")
            report.append("")
            
            # Average model performances
            avg_lgb = np.mean([r['individual_predictions']['lightgbm'] for r in valid_results])
            avg_cat = np.mean([r['individual_predictions']['catboost'] for r in valid_results])
            avg_ada = np.mean([r['individual_predictions']['adaboost'] for r in valid_results])
            
            report.append("AVERAGE MODEL PREDICTIONS")
            report.append("-" * 30)
            report.append(f"LightGBM average: {avg_lgb:.3f}")
            report.append(f"CatBoost average: {avg_cat:.3f}")
            report.append(f"AdaBoost average: {avg_ada:.3f}")
            report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS")
        report.append("-" * 30)
        
        for i, result in enumerate(valid_results, 1):
            report.append(f"\n{i}. {os.path.basename(result['image_path'])}")
            report.append(f"   Prediction: {result['prediction_label']}")
            report.append(f"   Individual Models:")
            report.append(f"     - LightGBM: {result['individual_predictions']['lightgbm']:.3f}")
            report.append(f"     - CatBoost: {result['individual_predictions']['catboost']:.3f}")
            report.append(f"     - AdaBoost: {result['individual_predictions']['adaboost']:.3f}")
        
        # Error results
        if error_results:
            report.append(f"\n\nERRORS ENCOUNTERED")
            report.append("-" * 30)
            for i, result in enumerate(error_results, 1):
                report.append(f"{i}. {result.get('image_path', 'Unknown')}: {result['error']}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {save_path}")
        
        print("\n" + report_text)
        return report_text
    
   

# Example usage and main function
def main():
    """
    Main function demonstrating how to use the prediction interface
    """
    # Initialize the prediction interface
    predictor = StrokePredictionInterface(
        unet_model_path="best_unet_model.h5",
        ensemble_models_path="my_stroke_ensemble"  # without file extensions
    )
 
    # Example 1: Predict on a single image
    print("\n" + "="*50)
    print("SINGLE IMAGE PREDICTION")
    print("="*50)
    
    single_image_path = "OIP (1).webp"
    result = predictor.predict_single_image(single_image_path)
    
    if "error" not in result:
        print(f"Image: {os.path.basename(result['image_path'])}")
        print(f"Prediction: {result['prediction_label']}")
        print(f"Stroke Probability: {result['stroke_probability']:.3f}")
    else:
        print(f"Error: {result['error']}")

    # Example 2: Predict on multiple images from a folder
    print("\n" + "="*50)
    print("BATCH PREDICTION")
    print("="*50)
    
    # Get all images from a folder
    test_folder = "test/stroke"
    if os.path.exists(test_folder):
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_paths = []
        
        for root, dirs, files in os.walk(test_folder):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))
        
        if image_paths:
            # Predict on all images
            results = predictor.predict_multiple_images(image_paths[:10])  # Limit to first 10 for demo
            
            # Generate visualizations and reports
            predictor.visualize_results(results, save_path="PredictionResults.png")
            predictor.generate_report(results, save_path="PredictionReport.txt")
        else:
            print(f"No images found in {test_folder}")
    else:
        print(f"Test folder {test_folder} does not exist")
    
main()