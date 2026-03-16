"""
Deep Learning Model Training Script
Trains CNN on mel spectrograms for deepfake detection
FIXED VERSION - Adds proper model saving and verification
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

from utils.spectrogram_generator import SpectrogramGenerator


class CNNModelTrainer:
    
    def __init__(self, input_shape=(128, 128, 1)):
        self.input_shape = input_shape
        self.model = None
        self.spec_generator = SpectrogramGenerator(sr=22050, n_mels=128)
    
    def build_cnn_model(self):
        """
        Build CNN architecture for spectrogram classification
        
        Returns:
            model: Compiled Keras model
        """
        model = models.Sequential([
            
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        return model
    
    def load_data(self, real_audio_dir, fake_audio_dir, target_shape=(128, 128)):
        """
        Load audio files and generate spectrograms
        
        Args:
            real_audio_dir: Directory containing real audio files
            fake_audio_dir: Directory containing fake audio files
            target_shape: Target spectrogram shape
            
        Returns:
            X: Spectrogram array
            y: Labels (0=fake, 1=real)
        """
        print("Loading real audio files...")
        real_files = [os.path.join(real_audio_dir, f) for f in os.listdir(real_audio_dir) 
                      if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]
        
        print("Loading fake audio files...")
        fake_files = [os.path.join(fake_audio_dir, f) for f in os.listdir(fake_audio_dir) 
                      if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]
        
        if len(real_files) == 0:
            raise ValueError(f"No audio files found in {real_audio_dir}")
        if len(fake_files) == 0:
            raise ValueError(f"No audio files found in {fake_audio_dir}")
        
        
        real_labels = [1] * len(real_files)
        fake_labels = [0] * len(fake_files)
        
        
        all_files = real_files + fake_files
        all_labels = real_labels + fake_labels
        
        print(f"Total files: {len(all_files)} (Real: {len(real_files)}, Fake: {len(fake_files)})")
        print("Generating spectrograms... (this may take a while)")
        
        
        X, y = self.spec_generator.batch_generate(all_files, all_labels, target_shape)
        
        if len(X) == 0:
            raise ValueError("No spectrograms generated! Check your audio files.")
        
        print(f"Spectrogram shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        return X, y
    
    def train(self, X, y, test_size=0.2, validation_split=0.2, epochs=50, batch_size=32):
        """
        Train CNN model
        
        Args:
            X: Spectrogram array
            y: Labels
            test_size: Proportion of test set
            validation_split: Proportion of validation set from training data
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            history: Training history
            metrics: Dictionary of evaluation metrics
        """
        
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        
        self.model = self.build_cnn_model()
        
        print("\nModel Architecture:")
        self.model.summary()
        
        
        os.makedirs('models/dl_model', exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/dl_model/best_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        
        print("\nTraining CNN model...")
        history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        
        print("\nEvaluating on test set...")
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        print("\n" + "="*50)
        print("DEEP LEARNING MODEL EVALUATION")
        print("="*50)
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        
        print(f"\nSample predictions (first 5):")
        for i in range(min(5, len(y_test))):
            print(f"  True: {y_test[i]}, Pred: {y_pred[i]}, Proba: {y_pred_prob[i][0]:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        metrics = {
            'accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        return history, metrics
    
    def save_model(self, model_dir='models/dl_model'):
        """Save trained model with PROPER verification"""
        os.makedirs(model_dir, exist_ok=True)
        
        if self.model is None:
            raise ValueError("❌ Model is None - cannot save! Train the model first.")
        
        
        keras_path = os.path.join(model_dir, 'cnn_model.keras')
        try:
            print(f"\nSaving model to {keras_path}...")
            self.model.save(keras_path, save_format='keras')
            
            
            if not os.path.exists(keras_path):
                raise Exception("Model file was not created!")
            
            file_size = os.path.getsize(keras_path)
            if file_size < 1000000:  
                raise Exception(f"Model file is too small ({file_size} bytes) - likely corrupted!")
            
            print(f"✓ Model saved to: {keras_path}")
            print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            print(f"❌ Failed to save .keras model: {e}")
            raise
        
        
        h5_path = os.path.join(model_dir, 'cnn_model.h5')
        try:
            print(f"\nSaving model to {h5_path}...")
            self.model.save(h5_path, save_format='h5')
            
            file_size_h5 = os.path.getsize(h5_path)
            print(f"✓ Model also saved to: {h5_path}")
            print(f"  File size: {file_size_h5 / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            print(f"⚠ Could not save .h5 format: {e}")
            print("  (This is OK - .keras format is preferred)")
        
        
        print("\nValidating saved model...")
        try:
            test_model = keras.models.load_model(keras_path)
            
            
            test_spec = np.random.rand(1, 128, 128, 1)
            test_pred = test_model.predict(test_spec, verbose=0)
            
            print(f"✓ Model loaded and tested successfully!")
            print(f"  Test prediction: {test_pred[0][0]:.4f}")
            
        except Exception as e:
            raise Exception(f"❌ Model validation failed: {e}")
    
    def predict(self, audio_path):
        """
        Predict single audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            prediction: 0 (fake) or 1 (real)
            probability: Confidence score
        """
        
        
        mel_spec = self.spec_generator.generate_melspectrogram(audio_path)
        if mel_spec is None:
            return None, None
        
        
        spec_processed = self.spec_generator.prepare_for_cnn(mel_spec)
        spec_processed = np.expand_dims(spec_processed, axis=0)
        
        
        probability = self.model.predict(spec_processed, verbose=0)[0][0]
        prediction = 1 if probability > 0.5 else 0
        
        return prediction, probability


def main():
    
    print("="*50)
    print("DEEP LEARNING MODEL TRAINING")
    print("="*50)
    
    
    trainer = CNNModelTrainer(input_shape=(128, 128, 1))
    
    
    real_dir = 'data/train/real'
    fake_dir = 'data/train/fake'
    
    
    if not os.path.exists(real_dir):
        print(f"\n❌ ERROR: Directory not found: {real_dir}")
        print(f"Please create this directory and add real audio files")
        return
    
    if not os.path.exists(fake_dir):
        print(f"\n❌ ERROR: Directory not found: {fake_dir}")
        print(f"Please create this directory and add fake audio files")
        return
    
    try:
        
        X, y = trainer.load_data(real_dir, fake_dir)
        
        
        history, metrics = trainer.train(X, y, epochs=50, batch_size=32)
        
        
        trainer.save_model()
        
        print("\n" + "="*50)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()