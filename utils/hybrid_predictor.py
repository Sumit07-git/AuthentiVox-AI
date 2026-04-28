import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from utils.feature_extractor import AudioFeatureExtractor
from utils.spectrogram_generator import SpectrogramGenerator

_DL_CLIP_DURATION = 3


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
        try:
            if os.path.exists(ml_model_path):
                self.ml_model = joblib.load(ml_model_path)
                print(f"✓ ML model loaded  ({ml_model_path})")
                print(f"  → classes_: {self.ml_model.classes_}")
            else:
                print(f"⚠ ML model not found: {ml_model_path}")
        except Exception as e:
            print(f"⚠ ML model load error: {e}")

        try:
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(f"✓ Scaler loaded    ({scaler_path})")
            else:
                print(f"⚠ Scaler not found: {scaler_path}")
        except Exception as e:
            print(f"⚠ Scaler load error: {e}")

        paths_to_try = []
        if dl_model_path:
            paths_to_try.append(dl_model_path)
        for p in ['models/dl_model/best_model.keras',
                  'models/dl_model/cnn_model.keras',
                  'models/dl_model/best_model.h5',
                  'models/dl_model/cnn_model.h5']:
            if p not in paths_to_try:
                paths_to_try.append(p)

        dl_loaded = False
        for path in paths_to_try:
            if dl_loaded:
                break
            if not os.path.exists(path):
                continue
            try:
                size = os.path.getsize(path)
                if size < 10_000:
                    print(f"⚠ Skipping {path} — too small ({size} B, likely LFS pointer)")
                    continue
            except OSError as e:
                print(f"⚠ Cannot stat {path}: {e}")
                continue
            try:
                print(f"  Loading DL model from {path}…")
                self.dl_model = keras.models.load_model(path, compile=False)
                self.dl_model.compile(
                    optimizer=keras.optimizers.Adam(1e-3),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                print(f"✓ DL model loaded  ({path})")
                print(f"  → layers={len(self.dl_model.layers)}, input={self.dl_model.input_shape}")
                dl_loaded = True
            except Exception as e:
                print(f"⚠ Could not load {path}: {e}")
                self.dl_model = None

        if not dl_loaded:
            print("⚠ DL model unavailable — prediction will use ML only")

    def predict_ml(self, audio_path):
        if self.ml_model is None or self.scaler is None:
            return None, None
        try:
            features = self.feature_extractor.extract_features(audio_path)
            if features is None:
                return None, None

            feat_scaled = self.scaler.transform(features.reshape(1, -1))
            prediction = int(self.ml_model.predict(feat_scaled)[0])
            proba = self.ml_model.predict_proba(feat_scaled)[0]

            classes = list(self.ml_model.classes_)
            if 1 not in classes:
                print(f"⚠ Unexpected classes_: {classes}")
                return None, None
            real_idx = classes.index(1)
            p_real = float(proba[real_idx])
            return prediction, p_real

        except Exception as e:
            print(f"❌ ML prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def predict_dl(self, audio_path):
        if self.dl_model is None:
            return None, None
        try:
            mel = self.spec_generator.generate_melspectrogram(
                audio_path, duration=_DL_CLIP_DURATION)
            if mel is None:
                return None, None

            spec = self.spec_generator.prepare_for_cnn(mel)
            spec = np.expand_dims(spec, axis=0).astype(np.float32)

            p_real = float(self.dl_model.predict(spec, verbose=0)[0][0])
            prediction = 1 if p_real > 0.5 else 0
            return prediction, p_real

        except Exception as e:
            print(f"❌ DL prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def predict_hybrid(self, audio_path, method='weighted_average',
                       ml_weight=0.4, dl_weight=0.6):
        ml_pred, ml_p_real = self.predict_ml(audio_path)
        dl_pred, dl_p_real = self.predict_dl(audio_path)

        result = {
            'ml_prediction': ml_pred,
            'ml_confidence': ml_p_real,
            'dl_prediction': dl_pred,
            'dl_confidence': dl_p_real,
            'hybrid_prediction': None,
            'hybrid_confidence': None,
            'method': method,
        }

        if ml_pred is None and dl_pred is not None:
            result.update(
                hybrid_prediction=dl_pred,
                hybrid_confidence=dl_p_real if dl_pred == 1 else 1.0 - dl_p_real,
                method='dl_only'
            )
            return result

        if dl_pred is None and ml_pred is not None:
            result.update(
                hybrid_prediction=ml_pred,
                hybrid_confidence=ml_p_real if ml_pred == 1 else 1.0 - ml_p_real,
                method='ml_only'
            )
            return result

        if ml_pred is None and dl_pred is None:
            return result

        if method == 'weighted_average':
            hybrid_p_real = ml_weight * ml_p_real + dl_weight * dl_p_real
            hybrid_pred = 1 if hybrid_p_real > 0.5 else 0
            hybrid_conf = hybrid_p_real if hybrid_pred == 1 else 1.0 - hybrid_p_real
            result.update(hybrid_prediction=hybrid_pred, hybrid_confidence=hybrid_conf)

        elif method == 'voting':
            if ml_pred == dl_pred:
                avg_p = (ml_p_real + dl_p_real) / 2.0
                result.update(
                    hybrid_prediction=ml_pred,
                    hybrid_confidence=avg_p if ml_pred == 1 else 1.0 - avg_p
                )
            else:
                ml_c = ml_p_real if ml_pred == 1 else 1.0 - ml_p_real
                dl_c = dl_p_real if dl_pred == 1 else 1.0 - dl_p_real
                if ml_c >= dl_c:
                    result.update(hybrid_prediction=ml_pred, hybrid_confidence=ml_c)
                else:
                    result.update(hybrid_prediction=dl_pred, hybrid_confidence=dl_c)

        elif method == 'max_confidence':
            ml_c = ml_p_real if ml_pred == 1 else 1.0 - ml_p_real
            dl_c = dl_p_real if dl_pred == 1 else 1.0 - dl_p_real
            if ml_c >= dl_c:
                result.update(hybrid_prediction=ml_pred, hybrid_confidence=ml_c)
            else:
                result.update(hybrid_prediction=dl_pred, hybrid_confidence=dl_c)

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
            'method': result['method'],
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
    result = predictor.predict_hybrid(test_audio)
    formatted = predictor.format_result(result)
    print(f"\nML   : {formatted['ml_label']} ({formatted['ml_confidence_percent']})")
    print(f"DL   : {formatted['dl_label']} ({formatted['dl_confidence_percent']})")
    print(f"Final: {formatted['hybrid_label']} ({formatted['hybrid_confidence_percent']})")
    print(f"Method: {formatted['method']}")


if __name__ == "__main__":
    test_predictor()