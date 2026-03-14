"""
Machine Learning Model Training Script
Trains Random Forest classifier on extracted audio features
FIXED VERSION - Adds model verification
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_extractor import AudioFeatureExtractor


class MLModelTrainer:
    
    def __init__(self):
        self.feature_extractor = AudioFeatureExtractor(sr=22050, n_mfcc=13)
        self.model = None
        self.scaler = StandardScaler()
    
    def load_data(self, real_audio_dir, fake_audio_dir):
        """
        Load audio files and extract features
        
        Args:
            real_audio_dir: Directory containing real audio files
            fake_audio_dir: Directory containing fake audio files
            
        Returns:
            X: Feature matrix
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
        
        # Create labels
        real_labels = [1] * len(real_files)
        fake_labels = [0] * len(fake_files)
        
        # Combine
        all_files = real_files + fake_files
        all_labels = real_labels + fake_labels
        
        print(f"Total files: {len(all_files)} (Real: {len(real_files)}, Fake: {len(fake_files)})")
        print("Extracting features...")
        
        # Extract features
        X, y = self.feature_extractor.extract_batch_features(all_files, all_labels)
        
        if len(X) == 0:
            raise ValueError("No features extracted! Check your audio files.")
        
        print(f"Feature shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Sample feature values: {X[0][:5]}")  # Show first 5 values
        
        return X, y
    
    def train(self, X, y, test_size=0.2, optimize=False):
        """
        Train Random Forest classifier
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Proportion of test set
            optimize: Whether to perform hyperparameter tuning
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if optimize:
            print("Performing hyperparameter optimization...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, n_jobs=-1, verbose=1, scoring='accuracy'
            )
            grid_search.fit(X_train_scaled, y_train)
            
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            print("Training Random Forest...")
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*50)
        print("MACHINE LEARNING MODEL EVALUATION")
        print("="*50)
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"\nSample predictions (first 5):")
        for i in range(min(5, len(y_test))):
            print(f"  True: {y_test[i]}, Pred: {y_pred[i]}, Proba: {y_pred_proba[i]}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        metrics = {
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        return metrics
    
    def save_model(self, model_dir='models/ml_model'):
        """Save trained model and scaler with verification"""
        os.makedirs(model_dir, exist_ok=True)
        
        if self.model is None:
            raise ValueError("❌ Model is None - cannot save! Train the model first.")
        
        if self.scaler is None:
            raise ValueError("❌ Scaler is None - cannot save! Train the model first.")
        
        # Save model
        model_path = os.path.join(model_dir, 'rf_classifier.pkl')
        joblib.dump(self.model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        # Verify files were saved
        if not os.path.exists(model_path):
            raise Exception(f"❌ Model file not created: {model_path}")
        if not os.path.exists(scaler_path):
            raise Exception(f"❌ Scaler file not created: {scaler_path}")
        
        # Check file sizes
        model_size = os.path.getsize(model_path)
        scaler_size = os.path.getsize(scaler_path)
        
        print(f"\n✓ Model saved to: {model_path}")
        print(f"  File size: {model_size / 1024 / 1024:.2f} MB")
        
        print(f"✓ Scaler saved to: {scaler_path}")
        print(f"  File size: {scaler_size / 1024:.2f} KB")
        
        # Validate by loading
        print("\nValidating saved models...")
        try:
            test_model = joblib.load(model_path)
            test_scaler = joblib.load(scaler_path)
            
            # Test prediction
            test_features = np.random.rand(1, 32)
            test_scaled = test_scaler.transform(test_features)
            test_pred = test_model.predict(test_scaled)
            test_proba = test_model.predict_proba(test_scaled)
            
            print(f"✓ Models loaded and tested successfully!")
            print(f"  Test prediction: {test_pred[0]}, Proba: {test_proba[0]}")
            
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
        features = self.feature_extractor.extract_features(audio_path)
        if features is None:
            return None, None
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return prediction, probability


def main():
    
    print("="*50)
    print("MACHINE LEARNING MODEL TRAINING")
    print("="*50)
    
    # Initialize trainer
    trainer = MLModelTrainer()
    
    # Set data directories
    real_dir = 'data/train/real'
    fake_dir = 'data/train/fake'
    
    # Check directories exist
    if not os.path.exists(real_dir):
        print(f"\n❌ ERROR: Directory not found: {real_dir}")
        print(f"Please create this directory and add real audio files")
        return
    
    if not os.path.exists(fake_dir):
        print(f"\n❌ ERROR: Directory not found: {fake_dir}")
        print(f"Please create this directory and add fake audio files")
        return
    
    try:
        # Load data
        X, y = trainer.load_data(real_dir, fake_dir)
        
        # Train model
        metrics = trainer.train(X, y, optimize=False)
        
        # Save model
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