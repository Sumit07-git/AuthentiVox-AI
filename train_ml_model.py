import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
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

        print(
            f"Total files: {len(all_files)} "
            f"(Real: {len(real_files)}, Fake: {len(fake_files)})"
        )
        print("Extracting features...")

        X, y = self.feature_extractor.extract_batch_features(all_files, all_labels)

        if len(X) == 0:
            raise ValueError("No features extracted! Check your audio files.")

        print(f"Feature matrix shape : {X.shape}")
        print(f"Labels shape         : {y.shape}")
        print(f"Sample feature values (first 5 dims): {X[0][:5]}")

        return X, y

    def _check_class_balance(self, y):
        unique, counts = np.unique(y, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"\nClass distribution: {dist}")

        ratio = max(counts) / min(counts)
        if ratio > 3:
            print(
                f"⚠️  Imbalance ratio {ratio:.1f}:1 detected — "
                "enabling class_weight='balanced' and using F1-macro."
            )
            return 'balanced'

        print(f"✓ Classes roughly balanced (ratio {ratio:.1f}:1)")
        return None

    def train(self, X, y, test_size=0.2, optimize=False):
        class_weight = self._check_class_balance(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )
        print(f"\nTraining samples : {len(X_train)}")
        print(f"Test samples     : {len(X_test)}")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                random_state=42,
                class_weight=class_weight,
                n_jobs=-1
            ))
        ])

        if optimize:
            print("\nPerforming hyperparameter optimisation (Pipeline-safe)...")

            param_grid = {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }

            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=5,
                n_jobs=-1,
                verbose=1,
                scoring='f1_macro'
            )
            grid_search.fit(X_train, y_train)

            self.model = grid_search.best_estimator_
            print(f"Best params      : {grid_search.best_params_}")
            print(f"Best CV F1-macro : {grid_search.best_score_:.4f}")

        else:
            print("\nTraining Random Forest (Pipeline, no grid search)...")
            pipeline.set_params(
                classifier__n_estimators=200,
                classifier__max_depth=20,
                classifier__min_samples_split=5,
                classifier__min_samples_leaf=2
            )
            pipeline.fit(X_train, y_train)
            self.model = pipeline

        self.scaler = self.model.named_steps['scaler']

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average='macro')
        gap = abs(accuracy - f1_mac)

        print("\n" + "=" * 55)
        print("MODEL EVALUATION")
        print("=" * 55)
        print(f"Accuracy         : {accuracy:.4f}")
        print(f"F1-Macro         : {f1_mac:.4f}  ← primary metric")

        if gap > 0.10:
            print(
                f"\n🚨 SUS ALERT: Accuracy vs F1-Macro gap = {gap:.4f}\n"
                "   Your model is likely biased toward the majority class.\n"
                "   Suggestions:\n"
                "   • Collect more minority-class samples\n"
                "   • class_weight='balanced' is already ON — check your data\n"
                "   • Inspect per-class recall in the report below"
            )
        elif f1_mac < 0.70:
            print(
                f"\n⚠️  F1-Macro {f1_mac:.4f} is below 0.70.\n"
                "   Model needs improvement — try more data or better features."
            )
        else:
            print(f"\n✓ Report looks healthy (accuracy/F1-macro gap = {gap:.4f})")

        print(f"\nSample predictions (first 5):")
        for i in range(min(5, len(y_test))):
            print(
                f"  True: {y_test[i]}  "
                f"Pred: {y_pred[i]}  "
                f"Proba: {np.round(y_pred_proba[i], 3)}"
            )

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print(
            f"  TN={cm[0,0]}  FP={cm[0,1]}\n"
            f"  FN={cm[1,0]}  TP={cm[1,1]}"
        )

        return {
            'accuracy': accuracy,
            'f1_macro': f1_mac,
            'y_test': y_test,
            'y_pred': y_pred
        }

    def save_model(self, model_dir='models/ml_model'):
        os.makedirs(model_dir, exist_ok=True)

        if self.model is None:
            raise ValueError("❌ No model to save — run train() first.")

        model_path = os.path.join(model_dir, 'rf_pipeline.pkl')
        joblib.dump(self.model, model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ File not created: {model_path}")

        size_mb = os.path.getsize(model_path) / 1024 / 1024
        print(f"\n✓ Pipeline saved : {model_path} ({size_mb:.2f} MB)")

        print("Validating saved pipeline...")
        loaded = joblib.load(model_path)

        n_features = loaded.named_steps['scaler'].n_features_in_
        print(f"  Detected feature dimension : {n_features}")

        test_input = np.random.rand(1, n_features)
        pred = loaded.predict(test_input)
        proba = loaded.predict_proba(test_input)

        print(
            f"✓ Validation passed — "
            f"pred={pred[0]}, proba={np.round(proba[0], 3)}"
        )

    def predict(self, audio_path):
        if self.model is None:
            raise ValueError("❌ No model loaded — train or load a model first.")

        features = self.feature_extractor.extract_features(audio_path)
        if features is None:
            print(f"⚠️  Could not extract features from: {audio_path}")
            return None, None

        prediction = self.model.predict(features.reshape(1, -1))[0]
        probability = self.model.predict_proba(features.reshape(1, -1))[0]

        return prediction, probability

    def load_model(self, model_dir='models/ml_model'):
        model_path = os.path.join(model_dir, 'rf_pipeline.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found: {model_path}")

        self.model = joblib.load(model_path)
        self.scaler = self.model.named_steps['scaler']

        n_features = self.scaler.n_features_in_
        print(f"✓ Pipeline loaded from {model_path}")
        print(f"  Feature dimension : {n_features}")


def main():
    print("=" * 55)
    print("MACHINE LEARNING MODEL TRAINING")
    print("=" * 55)

    trainer = MLModelTrainer()
    real_dir = 'data/train/real'
    fake_dir = 'data/train/fake'

    missing = [d for d in (real_dir, fake_dir) if not os.path.exists(d)]
    if missing:
        for d in missing:
            print(f"\n❌ Directory not found: {d}")
            print("   Please create it and add audio files.")
        return

    try:
        X, y = trainer.load_data(real_dir, fake_dir)
        metrics = trainer.train(X, y, optimize=False)
        trainer.save_model()

        print("\n" + "=" * 55)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"   Accuracy : {metrics['accuracy']:.4f}")
        print(f"   F1-Macro : {metrics['f1_macro']:.4f}")
        print("=" * 55)

    except Exception as exc:
        print(f"\n❌ ERROR: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()