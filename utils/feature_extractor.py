"""
Audio Feature Extraction Utility - FIXED VERSION
Changes:
  1. Added peak normalization to handle microphone recordings
  2. Added silence detection (rejects near-silent clips)
  3. Added minimum duration check
"""

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
        """
        Peak-normalize audio to [-1, 1] range.
        Critical for microphone recordings which are quieter than studio audio.
        """
        peak = np.abs(y).max()
        if peak < 1e-8:
            return y
        return y / peak
    
    def extract_features(self, audio_path):
        """
        Extract comprehensive audio features with normalization.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            feature_vector: 1D numpy array of 32 features
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr, duration=30)
            
            # Check duration
            duration = librosa.get_duration(y=y, sr=sr)
            if duration < _MIN_DURATION_SEC:
                print(f"⚠ Skipping {audio_path}: too short ({duration:.2f}s)")
                return None
            
            # Check if near-silent (before normalization)
            rms_raw = float(np.mean(librosa.feature.rms(y=y)))
            if rms_raw < _MIN_RMS_ENERGY:
                print(f"⚠ Skipping {audio_path}: near-silent (RMS={rms_raw:.2e})")
                return None
            
            # ✅ FIX: Normalize loudness
            y = self._normalize_audio(y)
            
            # MFCC features (13 coefficients)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)  # 13 features
            mfcc_std = np.std(mfcc, axis=1)    # 13 features
            
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            
            # Temporal features
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
            rms_feature = rms_raw  # Keep original RMS as feature
            
            # Concatenate all features (13 + 13 + 3 + 3 = 32 features)
            feature_vector = np.concatenate([
                mfcc_mean,
                mfcc_std,
                [spectral_centroid, spectral_rolloff, spectral_bandwidth],
                [zcr, chroma, rms_feature]
            ])
            
            return feature_vector
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {str(e)}")
            return None
    
    def extract_batch_features(self, audio_paths, labels=None):
        """
        Extract features from multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            labels: Optional list of labels (0=fake, 1=real)
            
        Returns:
            X: Feature matrix (n_samples, 32)
            y: Labels (if provided)
        """
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