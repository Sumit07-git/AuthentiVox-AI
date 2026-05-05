"""
Hybrid Prediction Module - FIXED VERSION
Changes:
  1. Fixed ML/DL confidence calculation (p_real → confidence in prediction)
  2. Added audio normalization before DL prediction
  3. Increased DL clip duration 3s → 10s
  4. Fixed pipeline vs bare classifier detection
  5. Added robust classes_ index lookup
"""

import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from utils.feature_extractor import AudioFeatureExtractor
from utils.spectrogram_generator import SpectrogramGenerator

_DL_CLIP_DURATION = 10  # ✅ Increased from 3s to 10s


class HybridPredictor:
    """
    Hybrid predictor combining ML and DL models.
    """
    
    def __init__(self, ml_model_path=None, dl_model_path=None, scaler_path=None):
        """
        Initialize hybrid predictor.
        
        Args:
            ml_model_path: Path to ML model (pipeline or bare classifier)
            dl_model_path: Path to DL model (.keras or .h5)
            scaler_path: Path to scaler (only for bare classifier)
        """
        if ml_model_path is None:
            ml_model_path = 'models/ml_model/rf_pipeline.pkl'
        if scaler_path is None:
            scaler_path = 'models/ml_model/scaler.pkl'
        
        self.feature_extractor = AudioFeatureExtractor(sr=22050, n_mfcc=13)
        self.spec_generator = SpectrogramGenerator(sr=22050, n_mels=128)
        
        self.ml_model = None
        self.dl_model = None
        self.scaler = None
        
        self.load_models(ml_model_path, dl_model_path, scaler_path)
    
    def load_models(self, ml_model_path, dl_model_path, scaler_path):
        """Load trained models with comprehensive error handling."""
        
        # ========== ML Model ==========
        try:
            if os.path.exists(ml_model_path):
                self.ml_model = joblib.load(ml_model_path)
                print(f"✓ ML model loaded: {ml_model_path}")
                
                # Check if it's a pipeline
                if hasattr(self.ml_model, 'named_steps'):
                    clf = self.ml_model.named_steps.get('classifier')
                    if clf:
                        print(f"  → Classifier classes: {clf.classes_}")
                    scaler_step = self.ml_model.named_steps.get('scaler')
                    if scaler_step:
                        self.scaler = scaler_step
                        print(f"  → Scaler features: {scaler_step.n_features_in_}")
                else:
                    print("  → Bare classifier (not pipeline)")
            else:
                # Try legacy path
                legacy_path = 'models/ml_model/rf_classifier.pkl'
                if os.path.exists(legacy_path):
                    self.ml_model = joblib.load(legacy_path)
                    print(f"✓ ML model loaded (legacy): {legacy_path}")
                else:
                    print(f"⚠ ML model not found: {ml_model_path}")
        except Exception as e:
            print(f"⚠ Error loading ML model: {e}")
        
        # ========== Standalone Scaler (for bare classifier) ==========
        try:
            if scaler_path and os.path.exists(scaler_path) and self.scaler is None:
                self.scaler = joblib.load(scaler_path)
                print(f"✓ Scaler loaded: {scaler_path}")
        except Exception as e:
            print(f"⚠ Error loading scaler: {e}")
        
        # ========== DL Model ==========
        paths_to_try = []
        if dl_model_path:
            paths_to_try.append(dl_model_path)
        
        # Default paths
        default_paths = [
            'models/dl_model/best_model.keras',
            'models/dl_model/cnn_model.keras',
            'models/dl_model/best_model.h5',
            'models/dl_model/cnn_model.h5'
        ]
        
        for path in default_paths:
            if path not in paths_to_try:
                paths_to_try.append(path)
        
        dl_loaded = False
        for path in paths_to_try:
            if dl_loaded:
                break
            
            if not os.path.exists(path):
                continue
            
            # Check file size (skip LFS pointers)
            try:
                file_size = os.path.getsize(path)
                if file_size < 10000:  # < 10KB likely LFS pointer
                    print(f"⚠ Skipping {path}: too small ({file_size} bytes)")
                    continue
            except Exception as e:
                print(f"⚠ Cannot check size of {path}: {e}")
                continue
            
            # Try to load
            try:
                print(f"  Loading DL model: {path}...")
                self.dl_model = keras.models.load_model(path, compile=False)
                
                # Recompile
                self.dl_model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                print(f"✓ DL model loaded: {path}")
                print(f"  → Layers: {len(self.dl_model.layers)}")
                print(f"  → Input shape: {self.dl_model.input_shape}")
                dl_loaded = True
                
            except Exception as e:
                print(f"⚠ Failed to load {path}: {e}")
                self.dl_model = None
        
        if not dl_loaded:
            print("⚠ DL model not loaded - will use ML only")
    
    def _is_pipeline(self):
        """Check if ML model is a pipeline."""
        return hasattr(self.ml_model, 'named_steps')
    
    @staticmethod
    def _normalize_audio(y):
        """Peak-normalize audio to [-1, 1] range."""
        peak = np.abs(y).max()
        if peak < 1e-8:
            return y
        return y / peak
    
    def predict_ml(self, audio_path):
        """
        Predict using ML model only.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            prediction: 0 (fake) or 1 (real)
            p_real: Probability of being REAL [0, 1]
        """
        if self.ml_model is None:
            return None, None
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(audio_path)
            if features is None:
                return None, None
            
            feat_input = features.reshape(1, -1)
            
            # Predict
            if self._is_pipeline():
                # Pipeline handles scaling internally
                prediction = int(self.ml_model.predict(feat_input)[0])
                proba = self.ml_model.predict_proba(feat_input)[0]
                clf = self.ml_model.named_steps['classifier']
                classes = list(clf.classes_)
            else:
                # Bare classifier - need manual scaling
                if self.scaler is not None:
                    feat_input = self.scaler.transform(feat_input)
                prediction = int(self.ml_model.predict(feat_input)[0])
                proba = self.ml_model.predict_proba(feat_input)[0]
                classes = list(self.ml_model.classes_)
            
            # ✅ FIX: Get p_real from correct class index
            if 1 not in classes:
                print(f"⚠ Unexpected classes: {classes}")
                return None, None
            
            real_idx = classes.index(1)
            p_real = float(proba[real_idx])
            
            print(f"  ML → pred={prediction} p_real={p_real:.4f} classes={classes}")
            
            return prediction, p_real
            
        except Exception as e:
            print(f"❌ ML prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def predict_dl(self, audio_path):
        """
        Predict using DL model only.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            prediction: 0 (fake) or 1 (real)
            p_real: Probability of being REAL [0, 1]
        """
        if self.dl_model is None:
            return None, None
        
        try:
            import librosa
            
            # ✅ FIX: Load with longer duration (10s instead of 3s)
            y, sr = librosa.load(audio_path, sr=22050, duration=_DL_CLIP_DURATION)
            
            # ✅ FIX: Normalize loudness
            y = self._normalize_audio(y)
            
            # Check if audio has content
            rms = float(np.sqrt(np.mean(y ** 2)))
            if rms < 1e-5:
                print(f"  DL → skipping: near-silent (RMS={rms:.2e})")
                return None, None
            
            # Generate spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Prepare for CNN
            spec = self.spec_generator.prepare_for_cnn(mel_spec_db)
            spec = np.expand_dims(spec, axis=0).astype(np.float32)
            
            # Predict
            p_real = float(self.dl_model.predict(spec, verbose=0)[0][0])
            prediction = 1 if p_real > 0.5 else 0
            
            print(f"  DL → pred={prediction} p_real={p_real:.4f} (clip={_DL_CLIP_DURATION}s)")
            
            return prediction, p_real
            
        except Exception as e:
            print(f"❌ DL prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def predict_hybrid(self, audio_path, method='weighted_average', ml_weight=0.4, dl_weight=0.6):
        """
        Hybrid prediction combining both models.
        
        Args:
            audio_path: Path to audio file
            method: Combination method ('weighted_average', 'voting', 'max_confidence')
            ml_weight: Weight for ML model
            dl_weight: Weight for DL model
            
        Returns:
            result: Dictionary with all predictions and confidences
        """
        print(f"\n--- Hybrid Prediction: {os.path.basename(audio_path)} ---")
        
        ml_pred, ml_p_real = self.predict_ml(audio_path)
        dl_pred, dl_p_real = self.predict_dl(audio_path)
        
        result = {
            'ml_prediction': ml_pred,
            'ml_confidence': ml_p_real,  # p_real for ML
            'dl_prediction': dl_pred,
            'dl_confidence': dl_p_real,  # p_real for DL
            'hybrid_prediction': None,
            'hybrid_confidence': None,
            'method': method
        }
        
        # Handle missing models
        if ml_pred is None and dl_pred is not None:
            result['hybrid_prediction'] = dl_pred
            result['hybrid_confidence'] = dl_p_real
            result['method'] = 'dl_only'
            return result
        
        if dl_pred is None and ml_pred is not None:
            result['hybrid_prediction'] = ml_pred
            result['hybrid_confidence'] = ml_p_real
            result['method'] = 'ml_only'
            return result
        
        if ml_pred is None and dl_pred is None:
            return result
        
        # ========== Hybrid Combination ==========
        if method == 'weighted_average':
            # Weighted average of p_real
            hybrid_p_real = ml_weight * ml_p_real + dl_weight * dl_p_real
            hybrid_pred = 1 if hybrid_p_real > 0.5 else 0
            
            result['hybrid_prediction'] = hybrid_pred
            result['hybrid_confidence'] = hybrid_p_real
        
        elif method == 'voting':
            # Vote between models
            if ml_pred == dl_pred:
                # Agreement
                avg_p_real = (ml_p_real + dl_p_real) / 2.0
                result['hybrid_prediction'] = ml_pred
                result['hybrid_confidence'] = avg_p_real
            else:
                # Disagreement - use confidence
                ml_conf = ml_p_real if ml_pred == 1 else (1.0 - ml_p_real)
                dl_conf = dl_p_real if dl_pred == 1 else (1.0 - dl_p_real)
                
                if ml_conf >= dl_conf:
                    result['hybrid_prediction'] = ml_pred
                    result['hybrid_confidence'] = ml_p_real
                else:
                    result['hybrid_prediction'] = dl_pred
                    result['hybrid_confidence'] = dl_p_real
        
        elif method == 'max_confidence':
            # Pick model with higher confidence
            ml_conf = ml_p_real if ml_pred == 1 else (1.0 - ml_p_real)
            dl_conf = dl_p_real if dl_pred == 1 else (1.0 - dl_p_real)
            
            if ml_conf >= dl_conf:
                result['hybrid_prediction'] = ml_pred
                result['hybrid_confidence'] = ml_p_real
            else:
                result['hybrid_prediction'] = dl_pred
                result['hybrid_confidence'] = dl_p_real
        
        print(f"  Hybrid → pred={result['hybrid_prediction']} p_real={result['hybrid_confidence']:.4f} method={result['method']}")
        
        return result