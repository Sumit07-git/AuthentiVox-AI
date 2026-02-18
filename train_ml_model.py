"""
Machine Learning Model Training Script
Trains Random Forest classifier on extracted audio features
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
    """Train and evaluate Random Forest classifier"""
    
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
        print("Extracting features...")
        
        # Extract features
        X, y = self.feature_extractor.extract_batch_features(all_files, all_labels)
        
        print(f"Feature shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        return X, y
    
    def train(self, X, y, test_size=0.2, optimize=True):
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
            print("Training Random Forest with default parameters...")
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
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*50)
        print("MACHINE LEARNING MODEL EVALUATION")
        print("="*50)
        print(f"\nAccuracy: {accuracy:.4f}")
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
        """
        Save trained model and scaler
        
        Args:
            model_dir: Directory to save model
        """
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'rf_classifier.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"\nModel saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
    
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
    """Main training function"""
    print("="*50)
    print("MACHINE LEARNING MODEL TRAINING")
    print("="*50)
    
    # Initialize trainer
    trainer = MLModelTrainer()
    
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
    metrics = trainer.train(X, y, optimize=False)  # Set to True for hyperparameter tuning
    
    # Save model
    trainer.save_model()
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()