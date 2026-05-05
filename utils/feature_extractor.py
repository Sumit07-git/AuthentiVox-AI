import numpy as np
import librosa
import warnings

warnings.filterwarnings('ignore')

_MIN_DURATION_SEC = 0.5
_MIN_RMS_ENERGY = 1e-5


class AudioFeatureExtractor:
    def __init__(self, sr=22050, n_mfcc=13):
        self.sr = sr
        self.n_mfcc = n_mfcc

    @staticmethod
    def _normalize_audio(y):
        peak = np.abs(y).max()
        if peak < 1e-8:
            return y
        return y / peak

    def extract_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, duration=30)

            duration = librosa.get_duration(y=y, sr=sr)
            if duration < _MIN_DURATION_SEC:
                print(f"⚠ Skipping {audio_path}: too short ({duration:.2f}s)")
                return None

            rms_raw = float(np.mean(librosa.feature.rms(y=y)))
            if rms_raw < _MIN_RMS_ENERGY:
                print(f"⚠ Skipping {audio_path}: near-silent (RMS={rms_raw:.2e})")
                return None

            y = self._normalize_audio(y)

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)

            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
            rms_feature = rms_raw

            feature_vector = np.concatenate([
                mfcc_mean,
                mfcc_std,
                [spectral_centroid, spectral_rolloff, spectral_bandwidth],
                [zcr, chroma, rms_feature],
            ])

            return feature_vector

        except Exception as e:
            print(f"Error extracting features from {audio_path}: {str(e)}")
            return None

    def extract_batch_features(self, audio_paths, labels=None):
        features = []
        valid_labels = []

        for i, path in enumerate(audio_paths):
            feature = self.extract_features(path)
            if feature is not None:
                features.append(feature)
                if labels is not None:
                    valid_labels.append(labels[i])

        X = np.array(features)

        if labels is not None:
            y = np.array(valid_labels)
            return X, y

        return X