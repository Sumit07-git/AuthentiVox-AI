"""
Audio Feature Extraction Utility — Fixed Version
Fix: Added minimum duration and RMS energy check to reject silent/too-short clips
     that would produce unreliable MFCC statistics and corrupt ML predictions.
"""

import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')

_MIN_DURATION_SEC = 0.5   # reject clips shorter than this
_MIN_RMS_ENERGY   = 1e-5  # reject near-silent clips below this RMS


class AudioFeatureExtractor:

    def __init__(self, sr=22050, n_mfcc=13):
        self.sr     = sr
        self.n_mfcc = n_mfcc

    def extract_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, duration=30)

            # ✅ FIX: Reject clips that are too short
            duration = librosa.get_duration(y=y, sr=sr)
            if duration < _MIN_DURATION_SEC:
                print(f"⚠ Skipping {audio_path}: too short ({duration:.2f}s < {_MIN_DURATION_SEC}s)")
                return None

            # ✅ FIX: Reject near-silent clips (RMS energy check)
            rms_energy = float(np.mean(librosa.feature.rms(y=y)))
            if rms_energy < _MIN_RMS_ENERGY:
                print(f"⚠ Skipping {audio_path}: near-silent (RMS={rms_energy:.2e})")
                return None

            # MFCC features
            mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std  = np.std(mfcc,  axis=1)

            # Spectral features
            spectral_centroid   = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff    = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            spectral_bandwidth  = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

            # Temporal features
            zcr    = np.mean(librosa.feature.zero_crossing_rate(y))
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))

            feature_vector = np.concatenate([
                mfcc_mean,
                mfcc_std,
                [spectral_centroid, spectral_rolloff, spectral_bandwidth],
                [zcr, chroma, rms_energy],
            ])

            return feature_vector

        except Exception as e:
            print(f"Error extracting features from {audio_path}: {str(e)}")
            return None

    def extract_batch_features(self, audio_paths, labels=None):
        features, valid_labels = [], []
        for i, path in enumerate(audio_paths):
            feature = self.extract_features(path)
            if feature is not None:
                features.append(feature)
                if labels is not None:
                    valid_labels.append(labels[i])

        X = np.array(features)
        if labels is not None:
            return X, np.array(valid_labels)
        return X