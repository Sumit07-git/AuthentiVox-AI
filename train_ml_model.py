import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_extractor import AudioFeatureExtractor


class MLModelTrainer:
    def __init__(self):
        self.feature_extractor = AudioFeatureExtractor(sr=22050, n_mfcc=13)
        self.model = None
        self.scaler = None

    def load_data(self, real_audio_dir, fake_audio_dir):
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
        unique, counts = np.unique(y, return_counts=True)
        ratio = max(counts) / min(counts)

        if ratio > 3:
            print(f"\n⚠ Imbalance detected (ratio {ratio:.1f}:1) — using class_weight='balanced'")
            class_weight = 'balanced'
        else:
            print(f"\n✓ Classes balanced (ratio {ratio:.1f}:1)")
            class_weight = None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"\nTrain samples: {len(X_train)}")
        print(f"Test samples:  {len(X_test)}")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=8,
                min_samples_leaf=4,
                max_features='sqrt',
                random_state=42,
                class_weight=class_weight,
                n_jobs=-1,
                oob_score=True,
            ))
        ])

        print("\nTraining Random Forest (Pipeline)...")
        pipeline.fit(X_train, y_train)

        self.model = pipeline
        self.scaler = pipeline.named_steps['scaler']

        clf = pipeline.named_steps['classifier']

        if hasattr(clf, 'oob_score_'):
            print(f"\n✓ OOB Score (generalization estimate): {clf.oob_score_:.4f}")
            if clf.oob_score_ < 0.80:
                print("  ⚠ OOB score is low — model may be underfitting or data quality is poor.")

        print("\nRunning 5-fold stratified cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        print(f"  CV AUC scores: {np.round(cv_scores, 4)}")
        print(f"  CV AUC mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        if cv_scores.std() > 0.05:
            print("  ⚠ High variance across folds — model may be unstable (small dataset?)")

        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average='macro')

        print("\n" + "=" * 60)
        print("HELD-OUT TEST EVALUATION")
        print("=" * 60)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"F1-Macro:  {f1_mac:.4f}")

        if 1 in clf.classes_:
            real_idx = list(clf.classes_).index(1)
            p_real_test = y_pred_proba[:, real_idx]
            auc = roc_auc_score(y_test, p_real_test)
            print(f"ROC-AUC:   {auc:.4f}")

            print(f"\np_real distribution on test set:")
            print(f"  min={p_real_test.min():.3f}  max={p_real_test.max():.3f}  mean={p_real_test.mean():.3f}")

            if p_real_test.max() < 0.6:
                print("🚨 WARNING: Model never predicts REAL with high confidence — check labels!")
            elif p_real_test.min() > 0.4:
                print("🚨 WARNING: Model always predicts REAL — possible label issue!")
            else:
                print("✓ Prediction distribution looks healthy")

        train_acc = accuracy_score(y_train, pipeline.predict(X_train))
        test_acc = accuracy
        overfit_gap = train_acc - test_acc
        print(f"\nOverfit check — train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  gap={overfit_gap:.4f}")
        if overfit_gap > 0.10:
            print(
                "⚠ Overfitting detected (gap > 0.10). "
                "Try reducing max_depth further or collecting more data."
            )
        else:
            print("✓ Train/test accuracy gap is acceptable.")

        importances = clf.feature_importances_
        top_indices = np.argsort(importances)[::-1][:10]
        print("\nTop 10 feature importances:")
        for rank, idx in enumerate(top_indices):
            print(f"  {rank+1:2d}. Feature {idx:2d}: {importances[idx]:.4f}")

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
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'overfit_gap': overfit_gap,
            'y_test': y_test,
            'y_pred': y_pred,
        }

    def save_model(self, model_dir='models/ml_model'):
        os.makedirs(model_dir, exist_ok=True)

        if self.model is None:
            raise ValueError("No model to save — train first!")

        pipeline_path = os.path.join(model_dir, 'rf_pipeline.pkl')
        joblib.dump(self.model, pipeline_path)

        size_mb = os.path.getsize(pipeline_path) / 1024 / 1024
        print(f"\n✓ Pipeline saved: {pipeline_path} ({size_mb:.2f} MB)")

        alias_path = os.path.join(model_dir, 'rf_classifier.pkl')
        joblib.dump(self.model, alias_path)
        print(f"✓ Alias saved:    {alias_path}")

        if self.scaler is not None:
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            print(f"✓ Scaler saved:   {scaler_path}")

        print("\nValidating saved pipeline...")
        loaded = joblib.load(pipeline_path)
        clf = loaded.named_steps['classifier']
        n_features = loaded.named_steps['scaler'].n_features_in_
        print(f"  Features:  {n_features}")
        print(f"  Classes:   {clf.classes_}")

        test_input = np.random.rand(1, n_features)
        pred = loaded.predict(test_input)
        proba = loaded.predict_proba(test_input)
        print(f"  Test: pred={pred[0]} proba={np.round(proba[0], 3)}")
        print("✓ Validation successful")

    def main():
        print("=" * 60)
        print("MACHINE LEARNING MODEL TRAINING (FULLY FIXED)")
        print("=" * 60)

        trainer = MLModelTrainer()
        real_dir = 'data/train/real'
        fake_dir = 'data/train/fake'

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
            print(f"   Accuracy:    {metrics['accuracy']:.4f}")
            print(f"   F1-Macro:    {metrics['f1_macro']:.4f}")
            print(f"   CV AUC:      {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
            print(f"   Overfit gap: {metrics['overfit_gap']:.4f}")
            print("=" * 60)

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()