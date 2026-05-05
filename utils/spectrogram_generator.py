"""
Spectrogram Generation Utility - FIXED VERSION
Changes:
  1. Added epsilon guard in normalization to prevent NaN from silent audio
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class SpectrogramGenerator:
    
    def __init__(self, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def generate_melspectrogram(self, audio_path, duration=10):
        """
        Generate mel spectrogram from audio file.
        
        Args:
            audio_path: Path to audio file
            duration: Duration in seconds to process
            
        Returns:
            mel_spectrogram_db: 2D numpy array in dB scale
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, duration=duration)
            
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            return mel_spec_db
            
        except Exception as e:
            print(f"Error generating spectrogram for {audio_path}: {str(e)}")
            return None
    
    def prepare_for_cnn(self, mel_spec_db, target_shape=(128, 128)):
        """
        Prepare spectrogram for CNN input.
        
        Args:
            mel_spec_db: Mel spectrogram in dB
            target_shape: Target shape (height, width)
            
        Returns:
            processed_spec: Normalized and resized spectrogram (128, 128, 1)
        """
        # Pad or crop to target width
        if mel_spec_db.shape[1] < target_shape[1]:
            pad_width = target_shape[1] - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :target_shape[1]]
        
        # ✅ FIX: Guard against division by zero (silent audio)
        spec_min = mel_spec_db.min()
        spec_max = mel_spec_db.max()
        denom = spec_max - spec_min
        
        if denom < 1e-8:
            # Silent audio - return zeros instead of NaN
            mel_spec_normalized = np.zeros_like(mel_spec_db)
        else:
            mel_spec_normalized = (mel_spec_db - spec_min) / denom
        
        # Add channel dimension
        mel_spec_normalized = np.expand_dims(mel_spec_normalized, axis=-1)
        
        return mel_spec_normalized
    
    def batch_generate(self, audio_paths, labels=None, target_shape=(128, 128)):
        """
        Generate spectrograms for multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            labels: Optional list of labels
            target_shape: Target shape for CNN
            
        Returns:
            X: Array of spectrograms
            y: Labels (if provided)
        """
        spectrograms = []
        valid_labels = []
        
        for i, path in enumerate(audio_paths):
            mel_spec = self.generate_melspectrogram(path)
            if mel_spec is not None:
                processed = self.prepare_for_cnn(mel_spec, target_shape)
                spectrograms.append(processed)
                if labels is not None:
                    valid_labels.append(labels[i])
        
        X = np.array(spectrograms)
        
        if labels is not None:
            y = np.array(valid_labels)
            return X, y
        
        return X