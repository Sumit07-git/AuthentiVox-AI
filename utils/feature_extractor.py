"""
Audio Feature Extraction Utility
Extracts MFCC, spectral, and temporal features from audio files
"""

import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    
    
    def __init__(self, sr=22050, n_mfcc=13):
        """
        Initialize feature extractor
        
        Args:
            sr: Sample rate for audio processing
            n_mfcc: Number of MFCC coefficients to extract
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
    
    def extract_features(self, audio_path):
        """
        Extract comprehensive audio features
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            feature_vector: 1D numpy array of features
        """
        try:
            
            y, sr = librosa.load(audio_path, sr=self.sr, duration=30)
            
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            
            
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            
            
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
            
            
            rms = np.mean(librosa.feature.rms(y=y))
            
            
            feature_vector = np.concatenate([
                mfcc_mean,
                mfcc_std,
                [spectral_centroid, spectral_rolloff, spectral_bandwidth],
                [zcr, chroma, rms]
            ])
            
            return feature_vector
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {str(e)}")
            return None
    
    def extract_batch_features(self, audio_paths, labels=None):
        """
        Extract features from multiple audio files
        
        Args:
            audio_paths: List of audio file paths
            labels: Optional list of labels (0=fake, 1=real)
            
        Returns:
            X: Feature matrix
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