"""
Deep Learning Model Training Script
Trains CNN on mel spectrograms for deepfake detection
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

# Fix import path - add both current dir and parent dir
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

from utils.spectrogram_generator import SpectrogramGenerator


class CNNModelTrainer:
    """Train and evaluate CNN for spectrogram classification"""
    
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
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output Layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
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
                      if f.endswith(('.wav', '.mp3', '.flac'))]
        
        print("Loading fake audio files...")
        fake_files = [os.path.join(fake_audio_dir, f) for f in os.listdir(fake_audio_dir) 
                      if f.endswith(('.wav', '.mp3', '.flac'))]
        
        # Create labels
        real_labels = [1] * len(real_files)
        fake_labels = [0] * len(fake_files)
        
        # Combine
        all_files = real_files + fake_files
        all_labels = real_labels + fake_labels
        
        print(f"Total files: {len(all_files)} (Real: {len(real_files)}, Fake: {len(fake_files)})")
        print("Generating spectrograms...")
        
        # Generate spectrograms
        X, y = self.spec_generator.batch_generate(all_files, all_labels, target_shape)
        
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
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Build model
        self.model = self.build_cnn_model()
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks
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
        
        # Train model
        print("\nTraining CNN model...")
        history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        print("\nEvaluating on test set...")
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        # Predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        print("\n" + "="*50)
        print("DEEP LEARNING MODEL EVALUATION")
        print("="*50)
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
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
        """
        Save trained model
        
        Args:
            model_dir: Directory to save model
        """
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'cnn_model.keras')
        self.model.save(model_path)
        
        print(f"\nModel saved to: {model_path}")
    
    def predict(self, audio_path):
        """
        Predict single audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            prediction: 0 (fake) or 1 (real)
            probability: Confidence score
        """
        # Generate spectrogram
        mel_spec = self.spec_generator.generate_melspectrogram(audio_path)
        if mel_spec is None:
            return None, None
        
        # Prepare for CNN
        spec_processed = self.spec_generator.prepare_for_cnn(mel_spec)
        spec_processed = np.expand_dims(spec_processed, axis=0)
        
        # Predict
        probability = self.model.predict(spec_processed, verbose=0)[0][0]
        prediction = 1 if probability > 0.5 else 0
        
        return prediction, probability


def main():
    """Main training function"""
    print("="*50)
    print("DEEP LEARNING MODEL TRAINING")
    print("="*50)
    
    # Initialize trainer
    trainer = CNNModelTrainer(input_shape=(128, 128, 1))
    
    # Set data directories
    real_dir = 'data/train/real'
    fake_dir = 'data/train/fake'
    
    # Check if directories exist
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print("\nERROR: Training data directories not found!")
        print(f"Please create and populate: {real_dir} and {fake_dir}")
        print("Add your real and fake audio samples (.wav or .mp3 files)")
        return
    
    # Load and prepare data
    X, y = trainer.load_data(real_dir, fake_dir)
    
    # Train model
    history, metrics = trainer.train(X, y, epochs=50, batch_size=32)
    
    # Save model
    trainer.save_model()
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()