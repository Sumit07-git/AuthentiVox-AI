import os
import numpy as np
import joblib
import librosa
import tensorflow as tf
from tensorflow import keras
from utils.feature_extractor import AudioFeatureExtractor
from utils.spectrogram_generator import SpectrogramGenerator

_DL_CLIP_DURATION = 10


class HybridPredictor:
    def __init__(self, ml_model_path=None, dl_model_path=None, scaler_path=None):
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
        try:
            if os.path.exists(ml_model_path):
                self.ml_model = joblib.load(ml_model_path)
                print(f"✓ ML model loaded: {ml_model_path}")
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
                legacy_path = 'models/ml_model/rf_classifier.pkl'
                if os.path.exists(legacy_path):
                    self.ml_model = joblib.load(legacy_path)
                    print(f"✓ ML model loaded (legacy): {legacy_path}")
                else:
                    print(f"⚠ ML model not found: {ml_model_path}")
        except Exception as e:
            print(f"⚠ Error loading ML model: {e}")

        try:
            if scaler_path and os.path.exists(scaler_path) and self.scaler is None:
                self.scaler = joblib.load(scaler_path)
                print(f"✓ Scaler loaded: {scaler_path}")
        except Exception as e:
            print(f"⚠ Error loading scaler: {e}")

        paths_to_try = []
        if dl_model_path:
            paths_to_try.append(dl_model_path)

        default_paths = [
            'models/dl_model/best_model.keras',
            'models/dl_model/cnn_model.keras',
            'models/dl_model/best_model.h5',
            'models/dl_model/cnn_model.h5',
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
            try:
                file_size = os.path.getsize(path)
                if file_size < 10_000:
                    print(f"⚠ Skipping {path}: too small ({file_size} bytes)")
                    continue
            except Exception as e:
                print(f"⚠ Cannot check size of {path}: {e}")
                continue
            try:
                print(f"  Loading DL model: {path}...")
                self.dl_model = keras.models.load_model(path, compile=False)
                self.dl_model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy'],
                )
                print(f"✓ DL model loaded: {path}")
                print(f"  → Layers: {len(self.dl_model.layers)}")
                print(f"  → Input shape: {self.dl_model.input_shape}")
                dl_loaded = True
            except Exception as e:
                print(f"⚠ Failed to load {path}: {e}")
                self.dl_model = None

        if not dl_loaded:
            print("⚠ DL model not loaded — will use ML only")

    def _is_pipeline(self):
        return hasattr(self.ml_model, 'named_steps')

    @staticmethod
    def _normalize_audio(y):
        peak = np.abs(y).max()
        if peak < 1e-8:
            return y
        return y / peak

    def predict_ml(self, audio_path):
        if self.ml_model is None:
            return None, None
        try:
            features = self.feature_extractor.extract_features(audio_path)
            if features is None:
                return None, None

            feat_input = features.reshape(1, -1)

            if self._is_pipeline():
                prediction = int(self.ml_model.predict(feat_input)[0])
                proba = self.ml_model.predict_proba(feat_input)[0]
                clf = self.ml_model.named_steps['classifier']
                classes = list(clf.classes_)
            else:
                if self.scaler is not None:
                    feat_input = self.scaler.transform(feat_input)
                prediction = int(self.ml_model.predict(feat_input)[0])
                proba = self.ml_model.predict_proba(feat_input)[0]
                classes = list(self.ml_model.classes_)

            if 1 not in classes:
                print(f"⚠ Unexpected ML classes: {classes}")
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
        if self.dl_model is None:
            return None, None
        try:
            y, sr = librosa.load(audio_path, sr=22050, duration=_DL_CLIP_DURATION)

            y = self._normalize_audio(y)

            rms = float(np.sqrt(np.mean(y ** 2)))
            if rms < 1e-5:
                print(f"  DL → skipping: near-silent (RMS={rms:.2e})")
                return None, None

            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512
            )

            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            spec = self.spec_generator.prepare_for_cnn(mel_spec_db, target_shape=(128, 128))

            expected_shape = tuple(self.dl_model.input_shape[1:])
            if spec.shape != expected_shape:
                print(f"  DL → shape mismatch: got {spec.shape}, expected {expected_shape}")
                return None, None

            spec_batch = np.expand_dims(spec, axis=0).astype(np.float32)

            p_real = float(self.dl_model.predict(spec_batch, verbose=0)[0][0])
            prediction = 1 if p_real > 0.5 else 0

            print(f"  DL → pred={prediction} p_real={p_real:.4f} (clip={_DL_CLIP_DURATION}s)")
            return prediction, p_real

        except Exception as e:
            print(f"❌ DL prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def predict_hybrid(
        self,
        audio_path,
        method='weighted_average',
        ml_weight=0.4,
        dl_weight=0.6,
    ):
        print(f"\n--- Hybrid Prediction: {os.path.basename(audio_path)} ---")

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

        if method == 'weighted_average':
            hybrid_p_real = ml_weight * ml_p_real + dl_weight * dl_p_real
            hybrid_pred = 1 if hybrid_p_real > 0.5 else 0
            result['hybrid_prediction'] = hybrid_pred
            result['hybrid_confidence'] = hybrid_p_real

        elif method == 'voting':
            if ml_pred == dl_pred:
                avg_p_real = (ml_p_real + dl_p_real) / 2.0
                result['hybrid_prediction'] = ml_pred
                result['hybrid_confidence'] = avg_p_real
            else:
                ml_conf = ml_p_real if ml_pred == 1 else (1.0 - ml_p_real)
                dl_conf = dl_p_real if dl_pred == 1 else (1.0 - dl_p_real)
                if ml_conf >= dl_conf:
                    result['hybrid_prediction'] = ml_pred
                    result['hybrid_confidence'] = ml_p_real
                else:
                    result['hybrid_prediction'] = dl_pred
                    result['hybrid_confidence'] = dl_p_real

        elif method == 'max_confidence':
            ml_conf = ml_p_real if ml_pred == 1 else (1.0 - ml_p_real)
            dl_conf = dl_p_real if dl_pred == 1 else (1.0 - dl_p_real)
            if ml_conf >= dl_conf:
                result['hybrid_prediction'] = ml_pred
                result['hybrid_confidence'] = ml_p_real
            else:
                result['hybrid_prediction'] = dl_pred
                result['hybrid_confidence'] = dl_p_real

        print(
            f"  Hybrid → pred={result['hybrid_prediction']} "
            f"p_real={result['hybrid_confidence']:.4f} "
            f"method={result['method']}"
        )
        return result