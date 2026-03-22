"""
Hybrid Prediction Module
Combines ML and DL models for enhanced deepfake detection
FIXED VERSION - Resolves deployment model loading issues
"""

import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from utils.feature_extractor import AudioFeatureExtractor
from utils.spectrogram_generator import SpectrogramGenerator


class HybridPredictor:
    """
    Hybrid predictor combining ML and DL models
    """
    
    def __init__(self, ml_model_path=None, dl_model_path=None, scaler_path=None):
        """
        Initialize hybrid predictor
        
        Args:
            ml_model_path: Path to ML model file
            dl_model_path: Path to DL model file
            scaler_path: Path to scaler file
        """
        # Set default paths
        if ml_model_path is None:
            ml_model_path = 'models/ml_model/rf_classifier.pkl'
        if scaler_path is None:
            scaler_path = 'models/ml_model/scaler.pkl'
        
        # Initialize feature extractors
        self.feature_extractor = AudioFeatureExtractor(sr=22050, n_mfcc=13)
        self.spec_generator = SpectrogramGenerator(sr=22050, n_mels=128)
        
        # Initialize models as None
        self.ml_model = None
        self.dl_model = None
        self.scaler = None
        
        # Load models
        self.load_models(ml_model_path, dl_model_path, scaler_path)
    
    def load_models(self, ml_model_path, dl_model_path, scaler_path):
        """
        Load trained models with comprehensive error handling
        
        Args:
            ml_model_path: Path to ML model
            dl_model_path: Path to DL model (can be None for auto-detection)
            scaler_path: Path to scaler
        """
        # Load ML model
        try:
            if os.path.exists(ml_model_path):
                self.ml_model = joblib.load(ml_model_path)
                print(f"✓ ML model loaded from {ml_model_path}")
            else:
                print(f"⚠ ML model not found at {ml_model_path}")
        except Exception as e:
            print(f"⚠ Error loading ML model: {str(e)}")
        
        # Load scaler
        try:
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(f"✓ Scaler loaded from {scaler_path}")
            else:
                print(f"⚠ Scaler not found at {scaler_path}")
        except Exception as e:
            print(f"⚠ Error loading scaler: {str(e)}")
        
        # Load DL model with multiple fallback options
        dl_loaded = False
        
        # Build list of paths to try
        paths_to_try = []
        
        if dl_model_path is not None:
            paths_to_try.append(dl_model_path)
        
        # Add common default paths
        default_paths = [
            'models/dl_model/best_model.keras',
            'models/dl_model/cnn_model.keras',
            'models/dl_model/best_model.h5',
            'models/dl_model/cnn_model.h5'
        ]
        
        for path in default_paths:
            if path not in paths_to_try:
                paths_to_try.append(path)
        
        # Try loading from each path
        for path in paths_to_try:
            if dl_loaded:
                break
            
            # Check if file exists
            if not os.path.exists(path):
                continue
            
            # Check file size (detect LFS pointers)
            try:
                file_size = os.path.getsize(path)
                if file_size < 10000:  # Less than 10KB likely a pointer
                    print(f"⚠ Skipping {path} - file too small ({file_size} bytes, likely LFS pointer)")
                    continue
            except Exception as e:
                print(f"⚠ Cannot check size of {path}: {e}")
                continue
            
            # Attempt to load the model
            try:
                print(f" Attempting to load DL model from {path}...")
                
                # CRITICAL FIX: Load without compiling to avoid weight issues
                self.dl_model = keras.models.load_model(path, compile=False)
                
                # Manually compile after loading
                self.dl_model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Verify model loaded correctly
                if len(self.dl_model.layers) > 0:
                    print(f"✓ DL model loaded successfully from {path}")
                    print(f"  → Model has {len(self.dl_model.layers)} layers")
                    print(f"  → Input shape: {self.dl_model.input_shape}")
                    dl_loaded = True
                else:
                    print(f"⚠ Model loaded but has no layers")
                    self.dl_model = None
                
            except Exception as e:
                print(f"⚠ Failed to load {path}: {str(e)}")
                self.dl_model = None
                continue
        
        # Final status
        if not dl_loaded:
            print(f"⚠ Warning: DL model could not be loaded")
            print(f"   Tried paths: {', '.join(paths_to_try[:4])}")
            print(f"   Will use ML-only prediction")
    
    def predict_ml(self, audio_path):
        """
        Predict using ML model only
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            prediction: 0 (fake) or 1 (real), or None if failed
            confidence: Confidence score [0, 1], or None if failed
        """
        if self.ml_model is None or self.scaler is None:
            return None, None
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(audio_path)
            if features is None:
                return None, None
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predict
            prediction = self.ml_model.predict(features_scaled)[0]
            probabilities = self.ml_model.predict_proba(features_scaled)[0]
            
            # Get confidence for predicted class
            confidence = probabilities[1] if prediction == 1 else probabilities[0]
            
            return int(prediction), float(confidence)
            
        except Exception as e:
            print(f"❌ ML prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def predict_dl(self, audio_path):
        """
        Predict using DL model only
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            prediction: 0 (fake) or 1 (real), or None if failed
            confidence: Confidence score [0, 1], or None if failed
        """
        if self.dl_model is None:
            return None, None
        
        try:
            # Generate mel spectrogram
            mel_spec = self.spec_generator.generate_melspectrogram(audio_path)
            if mel_spec is None:
                return None, None
            
            # Prepare for CNN
            spec_processed = self.spec_generator.prepare_for_cnn(mel_spec)
            spec_processed = np.expand_dims(spec_processed, axis=0)
            
            # Predict
            probability = self.dl_model.predict(spec_processed, verbose=0)[0][0]
            prediction = 1 if probability > 0.5 else 0
            
            # Get confidence
            confidence = float(probability) if prediction == 1 else float(1 - probability)
            
            return int(prediction), confidence
            
        except Exception as e:
            print(f"❌ DL prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def predict_hybrid(self, audio_path, method='weighted_average', ml_weight=0.4, dl_weight=0.6):
        """
        Hybrid prediction combining both models
        
        Args:
            audio_path: Path to audio file
            method: Combination method ('weighted_average', 'voting', 'max_confidence')
            ml_weight: Weight for ML model (default 0.4)
            dl_weight: Weight for DL model (default 0.6)
            
        Returns:
            result: Dictionary containing all predictions and confidences
        """
        # Get predictions from both models
        ml_pred, ml_conf = self.predict_ml(audio_path)
        dl_pred, dl_conf = self.predict_dl(audio_path)
        
        # Initialize result dictionary
        result = {
            'ml_prediction': ml_pred,
            'ml_confidence': ml_conf,
            'dl_prediction': dl_pred,
            'dl_confidence': dl_conf,
            'hybrid_prediction': None,
            'hybrid_confidence': None,
            'method': method
        }
        
        # Handle cases where one or both models are unavailable
        if ml_pred is None and dl_pred is not None:
            # Only DL model available
            result['hybrid_prediction'] = dl_pred
            result['hybrid_confidence'] = dl_conf
            result['method'] = 'dl_only'
            return result
            
        elif dl_pred is None and ml_pred is not None:
            # Only ML model available
            result['hybrid_prediction'] = ml_pred
            result['hybrid_confidence'] = ml_conf
            result['method'] = 'ml_only'
            return result
            
        elif ml_pred is None and dl_pred is None:
            # Both models failed
            return result
        
        # Both models available - combine predictions
        if method == 'weighted_average':
            # Convert predictions to probabilities for class 1 (real)
            ml_prob = ml_conf if ml_pred == 1 else (1 - ml_conf)
            dl_prob = dl_conf if dl_pred == 1 else (1 - dl_conf)
            
            # Weighted average
            hybrid_prob = (ml_weight * ml_prob) + (dl_weight * dl_prob)
            
            # Final prediction
            result['hybrid_prediction'] = 1 if hybrid_prob > 0.5 else 0
            result['hybrid_confidence'] = hybrid_prob if result['hybrid_prediction'] == 1 else (1 - hybrid_prob)
            
        elif method == 'voting':
            # Simple voting with confidence tiebreaker
            if ml_pred == dl_pred:
                result['hybrid_prediction'] = ml_pred
                result['hybrid_confidence'] = (ml_conf + dl_conf) / 2
            else:
                # Disagreement - use higher confidence
                if ml_conf > dl_conf:
                    result['hybrid_prediction'] = ml_pred
                    result['hybrid_confidence'] = ml_conf
                else:
                    result['hybrid_prediction'] = dl_pred
                    result['hybrid_confidence'] = dl_conf
        
        elif method == 'max_confidence':
            # Use prediction with highest confidence
            if ml_conf > dl_conf:
                result['hybrid_prediction'] = ml_pred
                result['hybrid_confidence'] = ml_conf
            else:
                result['hybrid_prediction'] = dl_pred
                result['hybrid_confidence'] = dl_conf
        
        return result
    
    def get_prediction_label(self, prediction):
        """
        Convert numerical prediction to label
        
        Args:
            prediction: 0 or 1
            
        Returns:
            label: 'FAKE' or 'REAL'
        """
        if prediction is None:
            return 'UNKNOWN'
        return 'REAL' if prediction == 1 else 'FAKE'
    
    def format_result(self, result):
        """
        Format prediction result for display
        
        Args:
            result: Dictionary from predict_hybrid
            
        Returns:
            formatted: Dictionary with formatted strings
        """
        formatted = {
            'ml_label': self.get_prediction_label(result['ml_prediction']),
            'ml_confidence_percent': f"{result['ml_confidence']*100:.2f}%" if result['ml_confidence'] is not None else 'N/A',
            'dl_label': self.get_prediction_label(result['dl_prediction']),
            'dl_confidence_percent': f"{result['dl_confidence']*100:.2f}%" if result['dl_confidence'] is not None else 'N/A',
            'hybrid_label': self.get_prediction_label(result['hybrid_prediction']),
            'hybrid_confidence_percent': f"{result['hybrid_confidence']*100:.2f}%" if result['hybrid_confidence'] is not None else 'N/A',
            'method': result['method']
        }
        
        return formatted


def test_predictor():
    """Test the predictor with a sample file"""
    print("="*60)
    print("TESTING HYBRID PREDICTOR")
    print("="*60)
    
    predictor = HybridPredictor()
    
    # Test audio path
    test_audio = 'data/test/sample.wav'
    
    if not os.path.exists(test_audio):
        print(f"\n⚠ Test audio not found: {test_audio}")
        print("Please provide a test audio file")
        return
    
    print(f"\n Testing with: {test_audio}")
    
    # Run prediction
    result = predictor.predict_hybrid(test_audio, method='weighted_average')
    formatted = predictor.format_result(result)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"ML Model:     {formatted['ml_label']} ({formatted['ml_confidence_percent']})")
    print(f"DL Model:     {formatted['dl_label']} ({formatted['dl_confidence_percent']})")
    print(f"Hybrid:       {formatted['hybrid_label']} ({formatted['hybrid_confidence_percent']})")
    print(f"Method:       {formatted['method']}")
    print("="*60)


if __name__ == "__main__":
    test_predictor()