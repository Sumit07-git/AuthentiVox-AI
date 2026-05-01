"""
Spectrogram Generation Utility — Fixed Version
Fix: Division-by-zero when audio is silent (max==min → NaN spectrogram → CNN garbage output).
     Added epsilon guard in prepare_for_cnn normalization.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class SpectrogramGenerator:

    def __init__(self, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
        self.sr         = sr
        self.n_mels     = n_mels
        self.n_fft      = n_fft
        self.hop_length = hop_length

    def generate_melspectrogram(self, audio_path, duration=5):
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, duration=duration)
            mel_spec    = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=self.n_mels,
                n_fft=self.n_fft, hop_length=self.hop_length,
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            return mel_spec_db
        except Exception as e:
            print(f"Error generating spectrogram for {audio_path}: {str(e)}")
            return None

    def save_spectrogram_image(self, mel_spec_db, save_path, with_axes=False):
        if with_axes:
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                mel_spec_db, sr=self.sr, hop_length=self.hop_length,
                x_axis='time', y_axis='mel', cmap='viridis',
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel Spectrogram')
            plt.tight_layout()
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
        else:
            fig, ax = plt.subplots(figsize=(12, 4))
            librosa.display.specshow(
                mel_spec_db, sr=self.sr, hop_length=self.hop_length,
                cmap='viridis', ax=ax,
            )
            ax.axis('off')
            ax.set_frame_on(False)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.margins(0, 0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(save_path, dpi=100, bbox_inches='tight',
                        pad_inches=0, facecolor='black')
            plt.close()

    def prepare_for_cnn(self, mel_spec_db, target_shape=(128, 128)):
        # Pad or crop to target width
        if mel_spec_db.shape[1] < target_shape[1]:
            pad_width = target_shape[1] - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :target_shape[1]]

        # ✅ FIX: guard against division by zero when audio is silent (max == min → NaN)
        spec_min = mel_spec_db.min()
        spec_max = mel_spec_db.max()
        denom    = spec_max - spec_min
        if denom < 1e-8:
            # Silent/near-silent audio: return zeros rather than NaN
            mel_spec_normalized = np.zeros_like(mel_spec_db)
        else:
            mel_spec_normalized = (mel_spec_db - spec_min) / denom

        return np.expand_dims(mel_spec_normalized, axis=-1)

    def batch_generate(self, audio_paths, labels=None, target_shape=(128, 128)):
        spectrograms, valid_labels = [], []
        for i, path in enumerate(audio_paths):
            mel_spec = self.generate_melspectrogram(path)
            if mel_spec is not None:
                spectrograms.append(self.prepare_for_cnn(mel_spec, target_shape))
                if labels is not None:
                    valid_labels.append(labels[i])
        X = np.array(spectrograms)
        if labels is not None:
            return X, np.array(valid_labels)
        return X