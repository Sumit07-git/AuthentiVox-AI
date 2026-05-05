"""
ML Model Trainer - FIXED VERSION
Changes:
  1. Always saves as pipeline (scaler + classifier)
  2. Proper class balance checking
  3. Label distribution verification
  4. Post-training validation
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_extractor import AudioFeatureExtractor


class MLModelTrainer:
    
    def __init__(self):
        self.feature_extractor = AudioFeatureExtractor(sr=22050, n_mfcc=13)
        self.model = None  # Will be a Pipeline
        self.scaler = None
    
    def load_data(self, real_audio_dir, fake_audio_dir):
        """Load and extract features from audio files."""
        
        AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg')
        
        print("Loading real audio files...")
        real_files = [
            os.path.join(real_audio_dir, f)
            for f in os.listdir(real_audio_dir)
            if f.lower().endswith(AUDIO_EXTENSIONS)
        ]
        
        print("Loading fake audio files...")
        fake_files = [
            os.path.join(fake_audio_dir, f)
            for f in os.listdir(fake_audio_dir)
            if f.lower().endswith(AUDIO_EXTENSIONS)
        ]
        
        if len(real_files) == 0:
            raise ValueError(f"No audio files found in {real_audio_dir}")
        if len(fake_files) == 0:
            raise ValueError(f"No audio files found in {fake_audio_dir}")
        
        # ✅ EXPLICIT: 1 = REAL, 0 = FAKE
        all_files = real_files + fake_files
        all_labels = [1] * len(real_files) + [0] * len(fake_files)
        
        print(f"\nTotal files: {len(all_files)}")
        print(f"  REAL (1): {len(real_files)} files")
        print(f"  FAKE (0): {len(fake_files)} files")
        print("\nExtracting features...")
        
        X, y = self.feature_extractor.extract_batch_features(all_files, all_labels)
        
        if len(X) == 0:
            raise ValueError("No features extracted! Check your audio files.")
        
        print(f"\nFeature matrix: {X.shape}")
        print(f"Labels:         {y.shape}")
        
        unique, counts = np.unique(y, return_counts=True)
        for cls, cnt in zip(unique, counts):
            label_name = "REAL" if cls == 1 else "FAKE"
            print(f"  class {cls} ({label_name}): {cnt} samples")
        
        return X, y
    
    def train(self, X, y, test_size=0.2):
        """Train Random Forest classifier with pipeline."""
        
        # Check class balance
        unique, counts = np.unique(y, return_counts=True)
        ratio = max(counts) / min(counts)
        
        if ratio > 3:
            print(f"\n⚠ Imbalance detected (ratio {ratio:.1f}:1)")
            print("  Using class_weight='balanced'")
            class_weight = 'balanced'
        else:
            print(f"\n✓ Classes balanced (ratio {ratio:.1f}:1)")
            class_weight = None
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"\nTrain samples: {len(X_train)}")
        print(f"Test samples:  {len(X_test)}")
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight=class_weight,
                n_jobs=-1
            ))
        ])
        
        print("\nTraining Random Forest (Pipeline)...")
        pipeline.fit(X_train, y_train)
        
        self.model = pipeline
        self.scaler = pipeline.named_steps['scaler']
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average='macro')
        
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"F1-Macro:  {f1_mac:.4f}")
        
        # Verify classifier learned both directions
        clf = pipeline.named_steps['classifier']
        print(f"\nClassifier classes: {clf.classes_}")
        
        if 1 in clf.classes_:
            real_idx = list(clf.classes_).index(1)
            p_real_test = y_pred_proba[:, real_idx]
            
            print(f"p_real stats:")
            print(f"  min={p_real_test.min():.3f}")
            print(f"  max={p_real_test.max():.3f}")
            print(f"  mean={p_real_test.mean():.3f}")
            
            if p_real_test.max() < 0.6:
                print("\n🚨 WARNING: Model never predicts REAL with confidence!")
                print("   Check your training data labels!")
            elif p_real_test.min() > 0.4:
                print("\n🚨 WARNING: Model always predicts REAL!")
                print("   Check your training data labels!")
            else:
                print("\n✓ Model predictions look healthy")
        
        print(f"\nSample predictions (first 5):")
        for i in range(min(5, len(y_test))):
            print(f"  True={y_test[i]} Pred={y_pred[i]} Proba={np.round(y_pred_proba[i], 3)}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake(0)', 'Real(1)']))
        
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print(f"  TN={cm[0,0]} FP={cm[0,1]}")
        print(f"  FN={cm[1,0]} TP={cm[1,1]}")
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_mac,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def save_model(self, model_dir='models/ml_model'):
        """Save trained pipeline."""
        
        os.makedirs(model_dir, exist_ok=True)
        
        if self.model is None:
            raise ValueError("No model to save - train first!")
        
        # Save pipeline
        pipeline_path = os.path.join(model_dir, 'rf_pipeline.pkl')
        joblib.dump(self.model, pipeline_path)
        
        size_mb = os.path.getsize(pipeline_path) / 1024 / 1024
        print(f"\n✓ Pipeline saved: {pipeline_path} ({size_mb:.2f} MB)")
        
        # Save alias for backward compatibility
        alias_path = os.path.join(model_dir, 'rf_classifier.pkl')
        joblib.dump(self.model, alias_path)
        print(f"✓ Alias saved:    {alias_path}")
        
        # Save standalone scaler
        if self.scaler is not None:
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            print(f"✓ Scaler saved:   {scaler_path}")
        
        # Validate
        print("\nValidating saved pipeline...")
        loaded = joblib.load(pipeline_path)
        
        clf = loaded.named_steps['classifier']
        n_features = loaded.named_steps['scaler'].n_features_in_
        
        print(f"  Features:  {n_features}")
        print(f"  Classes:   {clf.classes_}")
        
        # Test prediction
        test_input = np.random.rand(1, n_features)
        pred = loaded.predict(test_input)
        proba = loaded.predict_proba(test_input)
        
        print(f"  Test: pred={pred[0]} proba={np.round(proba[0], 3)}")
        print("✓ Validation successful")


def main():
    print("=" * 60)
    print("MACHINE LEARNING MODEL TRAINING")
    print("=" * 60)
    
    trainer = MLModelTrainer()
    
    real_dir = 'data/train/real'
    fake_dir = 'data/train/fake'
    
    # Check directories exist
    for d in [real_dir, fake_dir]:
        if not os.path.exists(d):
            print(f"\n❌ Directory not found: {d}")
            print("   Create it and add audio files!")
            return
    
    try:
        X, y = trainer.load_data(real_dir, fake_dir)
        metrics = trainer.train(X, y)
        trainer.save_model()
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETED!")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   F1-Macro:  {metrics['f1_macro']:.4f}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()