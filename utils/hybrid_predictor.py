"""
Hybrid Prediction Module
Combines ML and DL models for enhanced deepfake detection
"""

import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from utils.feature_extractor import AudioFeatureExtractor
from utils.spectrogram_generator import SpectrogramGenerator


class HybridPredictor:
    
    
    def __init__(self, ml_model_path=None, dl_model_path=None, scaler_path=None):
        """
        Initialize hybrid predictor
        
        Args:
            ml_model_path: Path to ML model file
            dl_model_path: Path to DL model file
            scaler_path: Path to scaler file
        """
        
        if ml_model_path is None:
            ml_model_path = 'models/ml_model/rf_classifier.pkl'
        if dl_model_path is None:
            dl_model_path = 'models/dl_model/cnn_model.keras'
        if scaler_path is None:
            scaler_path = 'models/ml_model/scaler.pkl'
        
        
        self.feature_extractor = AudioFeatureExtractor(sr=22050, n_mfcc=13)
        self.spec_generator = SpectrogramGenerator(sr=22050, n_mels=128)
        
        
        self.ml_model = None
        self.dl_model = None
        self.scaler = None
        
        self.load_models(ml_model_path, dl_model_path, scaler_path)
    
    def load_models(self, ml_model_path, dl_model_path, scaler_path):
        """Load trained models with better error handling"""
        try:
            # Load ML model
            if os.path.exists(ml_model_path):
                self.ml_model = joblib.load(ml_model_path)
                print(f"✓ ML model loaded from {ml_model_path}")
            else:
                print(f"⚠ Warning: ML model not found at {ml_model_path}")
            
            # Load Scaler
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(f"✓ Scaler loaded from {scaler_path}")
            else:
                print(f"⚠ Warning: Scaler not found at {scaler_path}")
            
            # Load DL model - TRY BOTH FORMATS
            dl_loaded = False
            
            # Try .keras format first (newer)
            keras_path = dl_model_path.replace('.h5', '.keras')
            if os.path.exists(keras_path):
                try:
                    self.dl_model = keras.models.load_model(keras_path, compile=False)
                    print(f"✓ DL model loaded from {keras_path}")
                    dl_loaded = True
                except Exception as e:
                    print(f"⚠ Failed to load .keras model: {str(e)}")
            
            # Try .h5 format if .keras failed
            if not dl_loaded and os.path.exists(dl_model_path):
                try:
                    self.dl_model = keras.models.load_model(dl_model_path, compile=False)
                    print(f"✓ DL model loaded from {dl_model_path}")
                    dl_loaded = True
                except Exception as e:
                    print(f"⚠ Failed to load .h5 model: {str(e)}")
            
            if not dl_loaded:
                print(f"⚠ Warning: DL model not found or failed to load")
                print(f"   Tried: {keras_path} and {dl_model_path}")
                print(f"   Will use ML-only prediction")
                
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def predict_ml(self, audio_path):
        """
        Predict using ML model
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            prediction: 0 (fake) or 1 (real)
            confidence: Confidence score [0, 1]
        """
        if self.ml_model is None or self.scaler is None:
            return None, None
        
        try:
            
            features = self.feature_extractor.extract_features(audio_path)
            if features is None:
                return None, None
            
            
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            prediction = self.ml_model.predict(features_scaled)[0]
            probabilities = self.ml_model.predict_proba(features_scaled)[0]
            
            
            confidence = probabilities[1] if prediction == 1 else probabilities[0]
            
            return int(prediction), float(confidence)
            
        except Exception as e:
            print(f"ML prediction error: {str(e)}")
            return None, None
    
    def predict_dl(self, audio_path):
        """
        Predict using DL model
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            prediction: 0 (fake) or 1 (real)
            confidence: Confidence score [0, 1]
        """
        if self.dl_model is None:
            return None, None
        
        try:
            
            mel_spec = self.spec_generator.generate_melspectrogram(audio_path)
            if mel_spec is None:
                return None, None
            
            
            spec_processed = self.spec_generator.prepare_for_cnn(mel_spec)
            spec_processed = np.expand_dims(spec_processed, axis=0)
            
            
            probability = self.dl_model.predict(spec_processed, verbose=0)[0][0]
            prediction = 1 if probability > 0.5 else 0
            
            
            confidence = float(probability) if prediction == 1 else float(1 - probability)
            
            return int(prediction), confidence
            
        except Exception as e:
            print(f"DL prediction error: {str(e)}")
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
            result: Dictionary containing predictions and confidences
        """
        
        ml_pred, ml_conf = self.predict_ml(audio_path)
        dl_pred, dl_conf = self.predict_dl(audio_path)
        
        result = {
            'ml_prediction': ml_pred,
            'ml_confidence': ml_conf,
            'dl_prediction': dl_pred,
            'dl_confidence': dl_conf,
            'hybrid_prediction': None,
            'hybrid_confidence': None,
            'method': method
        }
        
        
        if ml_pred is None and dl_pred is not None:
            result['hybrid_prediction'] = dl_pred
            result['hybrid_confidence'] = dl_conf
            result['method'] = 'dl_only'
            return result
        elif dl_pred is None and ml_pred is not None:
            result['hybrid_prediction'] = ml_pred
            result['hybrid_confidence'] = ml_conf
            result['method'] = 'ml_only'
            return result
        elif ml_pred is None and dl_pred is None:
            return result
        
        
        if method == 'weighted_average':
            
            
            ml_prob = ml_conf if ml_pred == 1 else (1 - ml_conf)
            dl_prob = dl_conf if dl_pred == 1 else (1 - dl_conf)
            
            
            hybrid_prob = (ml_weight * ml_prob) + (dl_weight * dl_prob)
            result['hybrid_prediction'] = 1 if hybrid_prob > 0.5 else 0
            result['hybrid_confidence'] = hybrid_prob if result['hybrid_prediction'] == 1 else (1 - hybrid_prob)
            
        elif method == 'voting':
            
            if ml_pred == dl_pred:
                result['hybrid_prediction'] = ml_pred
                result['hybrid_confidence'] = (ml_conf + dl_conf) / 2
            else:
                
                if ml_conf > dl_conf:
                    result['hybrid_prediction'] = ml_pred
                    result['hybrid_confidence'] = ml_conf
                else:
                    result['hybrid_prediction'] = dl_pred
                    result['hybrid_confidence'] = dl_conf
        
        elif method == 'max_confidence':
            
            if ml_conf > dl_conf:
                result['hybrid_prediction'] = ml_pred
                result['hybrid_confidence'] = ml_conf
            else:
                result['hybrid_prediction'] = dl_pred
                result['hybrid_confidence'] = dl_conf
        
        return result
    
    def get_prediction_label(self, prediction):
        """
        Convert prediction to human-readable label
        
        Args:
            prediction: 0 or 1
            
        Returns:
            label: 'FAKE' or 'REAL'
        """
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
            'ml_label': self.get_prediction_label(result['ml_prediction']) if result['ml_prediction'] is not None else 'N/A',
            'ml_confidence_percent': f"{result['ml_confidence']*100:.2f}%" if result['ml_confidence'] is not None else 'N/A',
            'dl_label': self.get_prediction_label(result['dl_prediction']) if result['dl_prediction'] is not None else 'N/A',
            'dl_confidence_percent': f"{result['dl_confidence']*100:.2f}%" if result['dl_confidence'] is not None else 'N/A',
            'hybrid_label': self.get_prediction_label(result['hybrid_prediction']) if result['hybrid_prediction'] is not None else 'N/A',
            'hybrid_confidence_percent': f"{result['hybrid_confidence']*100:.2f}%" if result['hybrid_confidence'] is not None else 'N/A',
            'method': result['method']
        }
        
        return formatted


def test_predictor():
    
    print("="*50)
    print("TESTING HYBRID PREDICTOR")
    print("="*50)
    
    predictor = HybridPredictor()
    
    
    test_audio = 'data/test/sample.wav'
    
    if not os.path.exists(test_audio):
        print(f"\nTest audio not found: {test_audio}")
        print("Please provide a test audio file")
        return
    
    print(f"\nTesting with: {test_audio}")
    
    result = predictor.predict_hybrid(test_audio, method='weighted_average')
    formatted = predictor.format_result(result)
    
    print("\nPrediction Results:")
    print(f"ML Model: {formatted['ml_label']} (Confidence: {formatted['ml_confidence_percent']})")
    print(f"DL Model: {formatted['dl_label']} (Confidence: {formatted['dl_confidence_percent']})")
    print(f"Hybrid: {formatted['hybrid_label']} (Confidence: {formatted['hybrid_confidence_percent']})")
    print(f"Method: {formatted['method']}")


if __name__ == "__main__":
    test_predictor()