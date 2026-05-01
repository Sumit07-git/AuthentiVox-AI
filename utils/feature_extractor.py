"""
Audio Feature Extraction Utility — Fixed Version
Fixes:
  1. Minimum duration and RMS energy check to reject silent/too-short clips
  2. ✅ NEW: Peak normalization before feature extraction
     Microphone recordings are often much quieter than ASVspoof studio audio.
     Without normalization, MFCC mean values are shifted and the ML model
     sees a distribution it was never trained on → misclassifies as FAKE.
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

    @staticmethod
    def _normalize_audio(y):
        """
        ✅ NEW: Peak-normalize audio signal to [-1, 1].
        This is critical for microphone recordings which are typically much
        quieter than ASVspoof bonafide audio. Without normalization:
          - MFCC means shift significantly (quieter → lower energy features)
          - Spectral centroid/bandwidth are unreliable at low amplitude
          - ML model, trained on normalized ASVspoof audio, misclassifies
            quiet recordings as FAKE because the feature distribution doesn't match.
        """
        peak = np.abs(y).max()
        if peak < 1e-8:
            return y  # silent — feature extractor will reject it via RMS check
        return y / peak

    def extract_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, duration=30)

            # Reject clips that are too short
            duration = librosa.get_duration(y=y, sr=sr)
            if duration < _MIN_DURATION_SEC:
                print(f"⚠ Skipping {audio_path}: too short ({duration:.2f}s < {_MIN_DURATION_SEC}s)")
                return None

            # Reject near-silent clips before normalization
            rms_raw = float(np.mean(librosa.feature.rms(y=y)))
            if rms_raw < _MIN_RMS_ENERGY:
                print(f"⚠ Skipping {audio_path}: near-silent (RMS={rms_raw:.2e})")
                return None

            # ✅ NEW: normalize loudness so microphone recordings match training distribution
            y = self._normalize_audio(y)

            # MFCC features
            mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std  = np.std(mfcc,  axis=1)

            # Spectral features
            spectral_centroid  = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff   = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

            # Temporal features
            zcr    = np.mean(librosa.feature.zero_crossing_rate(y))
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
            # Use post-normalization RMS (always ~1.0 after norm, so use raw for feature)
            rms_feature = rms_raw  # keep raw RMS as a discriminative feature

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