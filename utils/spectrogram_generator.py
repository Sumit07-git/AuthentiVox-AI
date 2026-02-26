"""
Spectrogram Generation Utility
Converts audio to mel spectrograms for CNN model
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class SpectrogramGenerator:
    
    
    def __init__(self, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
        """
        Initialize spectrogram generator
        
        Args:
            sr: Sample rate
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Number of samples between frames
        """
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def generate_melspectrogram(self, audio_path, duration=5):
        """
        Generate mel spectrogram from audio file
        
        Args:
            audio_path: Path to audio file
            duration: Duration in seconds to process
            
        Returns:
            mel_spectrogram: 2D numpy array
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
    
    def save_spectrogram_image(self, mel_spec_db, save_path, with_axes=False):
        """
        Save spectrogram as clean image without axes
        
        Args:
            mel_spec_db: Mel spectrogram in dB
            save_path: Path to save image
            with_axes: If True, include axes labels (default: False)
        """
        if with_axes:
            
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                mel_spec_db,
                sr=self.sr,
                hop_length=self.hop_length,
                x_axis='time',
                y_axis='mel',
                cmap='viridis'
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel Spectrogram')
            plt.tight_layout()
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
        else:
            
            fig, ax = plt.subplots(figsize=(12, 4))
            
            
            librosa.display.specshow(
                mel_spec_db,
                sr=self.sr,
                hop_length=self.hop_length,
                cmap='viridis',
                ax=ax
            )
            
            
            ax.axis('off')
            ax.set_frame_on(False)
            
            
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.margins(0, 0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            
            
            plt.savefig(save_path, 
                       dpi=100, 
                       bbox_inches='tight', 
                       pad_inches=0,
                       facecolor='black')
            plt.close()
    
    def prepare_for_cnn(self, mel_spec_db, target_shape=(128, 128)):
        """
        Prepare spectrogram for CNN input
        
        Args:
            mel_spec_db: Mel spectrogram in dB
            target_shape: Target shape (height, width)
            
        Returns:
            processed_spec: Normalized and resized spectrogram
        """
        
        if mel_spec_db.shape[1] < target_shape[1]:
            
            pad_width = target_shape[1] - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            
            mel_spec_db = mel_spec_db[:, :target_shape[1]]
        
        
        mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        
        mel_spec_normalized = np.expand_dims(mel_spec_normalized, axis=-1)
        
        return mel_spec_normalized
    
    def batch_generate(self, audio_paths, labels=None, target_shape=(128, 128)):
        """
        Generate spectrograms for multiple audio files
        
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