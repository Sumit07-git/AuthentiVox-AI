"""
Hybrid Prediction Module
Combines ML and DL models for enhanced deepfake detection
FIXED: Correct probability handling using classes_ ordering
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
        if ml_model_path is None:
            ml_model_path = 'models/ml_model/rf_classifier.pkl'
        if scaler_path is None:
            scaler_path = 'models/ml_model/scaler.pkl'

        self.feature_extractor = AudioFeatureExtractor(sr=22050, n_mfcc=13)
        self.spec_generator = SpectrogramGenerator(sr=22050, n_mels=128)

        self.ml_model = None
        self.dl_model = None
        self.scaler = None

        self.load_models(ml_model_path, dl_model_path, scaler_path)

    def load_models(self, ml_model_path, dl_model_path, scaler_path):
        # Load ML model
        try:
            if os.path.exists(ml_model_path):
                self.ml_model = joblib.load(ml_model_path)
                print(f"✓ ML model loaded from {ml_model_path}")
                # FIX: Log classes_ so we can verify label ordering at startup
                print(f"  → ML model classes_: {self.ml_model.classes_}")
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

        # Load DL model
        dl_loaded = False
        paths_to_try = []

        if dl_model_path is not None:
            paths_to_try.append(dl_model_path)

        default_paths = [
            'models/dl_model/best_model.keras',
            'models/dl_model/cnn_model.keras',
            'models/dl_model/best_model.h5',
            'models/dl_model/cnn_model.h5'
        ]
        for path in default_paths:
            if path not in paths_to_try:
                paths_to_try.append(path)

        for path in paths_to_try:
            if dl_loaded:
                break
            if not os.path.exists(path):
                continue

            try:
                file_size = os.path.getsize(path)
                if file_size < 10000:
                    print(f"⚠ Skipping {path} - file too small ({file_size} bytes, likely LFS pointer)")
                    continue
            except Exception as e:
                print(f"⚠ Cannot check size of {path}: {e}")
                continue

            try:
                print(f"Attempting to load DL model from {path}...")
                self.dl_model = keras.models.load_model(path, compile=False)
                self.dl_model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                if len(self.dl_model.layers) > 0:
                    print(f"✓ DL model loaded from {path}")
                    print(f"  → Layers: {len(self.dl_model.layers)}, Input: {self.dl_model.input_shape}")
                    dl_loaded = True
                else:
                    print(f"⚠ Model loaded but has no layers")
                    self.dl_model = None
            except Exception as e:
                print(f"⚠ Failed to load {path}: {str(e)}")
                self.dl_model = None

        if not dl_loaded:
            print(f"⚠ DL model could not be loaded — using ML-only prediction")

    def predict_ml(self, audio_path):
        """
        Predict using ML model only.

        Returns (prediction, p_real) where:
          prediction = 0 (fake) or 1 (real)
          p_real     = probability that the audio is REAL, in [0, 1]
        """
        if self.ml_model is None or self.scaler is None:
            return None, None

        try:
            features = self.feature_extractor.extract_features(audio_path)
            if features is None:
                return None, None

            features_scaled = self.scaler.transform(features.reshape(1, -1))
            prediction = int(self.ml_model.predict(features_scaled)[0])
            probabilities = self.ml_model.predict_proba(features_scaled)[0]

            # FIX: Use classes_ to find the index of class 1 (real).
            # sklearn sorts classes_ so for integer labels 0,1 it is always
            # [0, 1], but we check explicitly to be safe against retraining
            # accidents or version differences.
            classes = list(self.ml_model.classes_)
            if 1 not in classes:
                print(f"⚠ Unexpected classes_: {classes}")
                return None, None

            real_idx = classes.index(1)   # column index for P(real)
            p_real = float(probabilities[real_idx])

            return prediction, p_real

        except Exception as e:
            print(f"❌ ML prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def predict_dl(self, audio_path):
        """
        Predict using DL model only.

        Returns (prediction, p_real) where:
          prediction = 0 (fake) or 1 (real)
          p_real     = probability that the audio is REAL, in [0, 1]

        The CNN output neuron uses sigmoid activation, so its output IS P(real)
        because the model was trained with label 1 = real.
        """
        if self.dl_model is None:
            return None, None

        try:
            mel_spec = self.spec_generator.generate_melspectrogram(audio_path)
            if mel_spec is None:
                return None, None

            spec_processed = self.spec_generator.prepare_for_cnn(mel_spec)
            spec_processed = np.expand_dims(spec_processed, axis=0)

            # FIX: raw sigmoid output = P(real), since label 1 = real in training
            p_real = float(self.dl_model.predict(spec_processed, verbose=0)[0][0])
            prediction = 1 if p_real > 0.5 else 0

            return prediction, p_real

        except Exception as e:
            print(f"❌ DL prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def predict_hybrid(self, audio_path, method='weighted_average',
                       ml_weight=0.4, dl_weight=0.6):
        """
        Hybrid prediction combining both models.

        All internal probabilities are P(real).  The final hybrid_confidence
        is always the probability of the *predicted* class, so it reads as
        "how confident are we in this prediction" regardless of real/fake.
        """
        ml_pred, ml_p_real = self.predict_ml(audio_path)
        dl_pred, dl_p_real = self.predict_dl(audio_path)

        result = {
            'ml_prediction': ml_pred,
            # FIX: store raw P(real) so callers can do their own math
            'ml_confidence': ml_p_real,
            'dl_prediction': dl_pred,
            'dl_confidence': dl_p_real,
            'hybrid_prediction': None,
            'hybrid_confidence': None,
            'method': method
        }

        # --- Handle missing models ---
        if ml_pred is None and dl_pred is not None:
            hybrid_p_real = dl_p_real
            result['hybrid_prediction'] = dl_pred
            # confidence = P(predicted class)
            result['hybrid_confidence'] = dl_p_real if dl_pred == 1 else (1.0 - dl_p_real)
            result['method'] = 'dl_only'
            return result

        if dl_pred is None and ml_pred is not None:
            result['hybrid_prediction'] = ml_pred
            result['hybrid_confidence'] = ml_p_real if ml_pred == 1 else (1.0 - ml_p_real)
            result['method'] = 'ml_only'
            return result

        if ml_pred is None and dl_pred is None:
            # Both models failed — cannot predict
            return result

        # --- Both models available ---
        if method == 'weighted_average':
            # FIX: ml_p_real and dl_p_real are already P(real), no inversion needed
            hybrid_p_real = (ml_weight * ml_p_real) + (dl_weight * dl_p_real)
            hybrid_pred = 1 if hybrid_p_real > 0.5 else 0

            result['hybrid_prediction'] = hybrid_pred
            # confidence = probability of the winning class
            result['hybrid_confidence'] = hybrid_p_real if hybrid_pred == 1 else (1.0 - hybrid_p_real)

        elif method == 'voting':
            if ml_pred == dl_pred:
                result['hybrid_prediction'] = ml_pred
                avg_p_real = (ml_p_real + dl_p_real) / 2.0
                result['hybrid_confidence'] = avg_p_real if ml_pred == 1 else (1.0 - avg_p_real)
            else:
                # Tie-break by whichever model is more confident
                ml_conf = ml_p_real if ml_pred == 1 else (1.0 - ml_p_real)
                dl_conf = dl_p_real if dl_pred == 1 else (1.0 - dl_p_real)
                if ml_conf >= dl_conf:
                    result['hybrid_prediction'] = ml_pred
                    result['hybrid_confidence'] = ml_conf
                else:
                    result['hybrid_prediction'] = dl_pred
                    result['hybrid_confidence'] = dl_conf

        elif method == 'max_confidence':
            ml_conf = ml_p_real if ml_pred == 1 else (1.0 - ml_p_real)
            dl_conf = dl_p_real if dl_pred == 1 else (1.0 - dl_p_real)
            if ml_conf >= dl_conf:
                result['hybrid_prediction'] = ml_pred
                result['hybrid_confidence'] = ml_conf
            else:
                result['hybrid_prediction'] = dl_pred
                result['hybrid_confidence'] = dl_conf

        return result

    def get_prediction_label(self, prediction):
        if prediction is None:
            return 'UNKNOWN'
        return 'REAL' if prediction == 1 else 'FAKE'

    def format_result(self, result):
        def pct(v):
            return f"{v * 100:.2f}%" if v is not None else 'N/A'

        return {
            'ml_label': self.get_prediction_label(result['ml_prediction']),
            'ml_confidence_percent': pct(result['ml_confidence']),
            'dl_label': self.get_prediction_label(result['dl_prediction']),
            'dl_confidence_percent': pct(result['dl_confidence']),
            'hybrid_label': self.get_prediction_label(result['hybrid_prediction']),
            'hybrid_confidence_percent': pct(result['hybrid_confidence']),
            'method': result['method']
        }


def test_predictor():
    print("=" * 60)
    print("TESTING HYBRID PREDICTOR")
    print("=" * 60)

    predictor = HybridPredictor()
    test_audio = 'data/test/sample.wav'

    if not os.path.exists(test_audio):
        print(f"\n⚠ Test audio not found: {test_audio}")
        return

    result = predictor.predict_hybrid(test_audio, method='weighted_average')
    formatted = predictor.format_result(result)

    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"ML Model:  {formatted['ml_label']} ({formatted['ml_confidence_percent']})")
    print(f"DL Model:  {formatted['dl_label']} ({formatted['dl_confidence_percent']})")
    print(f"Hybrid:    {formatted['hybrid_label']} ({formatted['hybrid_confidence_percent']})")
    print(f"Method:    {formatted['method']}")
    print("=" * 60)


if __name__ == "__main__":
    test_predictor()