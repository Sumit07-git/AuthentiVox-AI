import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils.spectrogram_generator import SpectrogramGenerator

CLIP_DURATION = 10


def augment_spectrogram(spec):
    spec = spec.copy()

    noise = np.random.normal(0, 0.005, spec.shape)
    spec = np.clip(spec + noise, 0.0, 1.0)

    if np.random.rand() < 0.5:
        t_mask = np.random.randint(5, 20)
        t_start = np.random.randint(0, max(1, spec.shape[1] - t_mask))
        spec[:, t_start:t_start + t_mask, :] = 0.0

    if np.random.rand() < 0.5:
        f_mask = np.random.randint(3, 15)
        f_start = np.random.randint(0, max(1, spec.shape[0] - f_mask))
        spec[f_start:f_start + f_mask, :, :] = 0.0

    if np.random.rand() < 0.5:
        gain = np.random.uniform(0.8, 1.2)
        spec = np.clip(spec * gain, 0.0, 1.0)

    return spec


def augment_dataset(X, y, factor=1):
    X_aug, y_aug = [X], [y]
    for _ in range(factor):
        aug = np.array([augment_spectrogram(s) for s in X])
        X_aug.append(aug)
        y_aug.append(y)
    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)


def build_cnn_model(input_shape=(128, 128, 1), l2=1e-4):
    reg = regularizers.l2(l2)
    init = keras.initializers.he_normal()
    inp = keras.Input(shape=input_shape)

    x = layers.Conv2D(
        32,
        (3, 3),
        padding='same',
        kernel_regularizer=reg,
        kernel_initializer=init,
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(
        32,
        (3, 3),
        padding='same',
        kernel_regularizer=reg,
        kernel_initializer=init,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        padding='same',
        kernel_regularizer=reg,
        kernel_initializer=init,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(
        64,
        (3, 3),
        padding='same',
        kernel_regularizer=reg,
        kernel_initializer=init,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Conv2D(
        128,
        (3, 3),
        padding='same',
        kernel_regularizer=reg,
        kernel_initializer=init,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(
        128,
        (3, 3),
        padding='same',
        kernel_regularizer=reg,
        kernel_initializer=init,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.35)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, kernel_regularizer=reg, kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.50)(x)
    x = layers.Dense(64, kernel_regularizer=reg, kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.30)(x)

    out = layers.Dense(
        1,
        activation='sigmoid',
        kernel_initializer=keras.initializers.glorot_uniform(),
    )(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ],
    )
    return model


class CNNModelTrainer:
    def __init__(self, input_shape=(128, 128, 1)):
        self.input_shape = input_shape
        self.model = None
        self.spec_generator = SpectrogramGenerator(sr=22050, n_mels=128)
        self.clip_duration = CLIP_DURATION

    def load_data(self, real_audio_dir, fake_audio_dir, target_shape=(128, 128)):
        print(f"Using clip duration: {self.clip_duration}s (must match _DL_CLIP_DURATION in hybrid_predictor.py)")
        print("Loading real audio files...")
        real_files = [
            os.path.join(real_audio_dir, f)
            for f in os.listdir(real_audio_dir)
            if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))
        ]
        print("Loading fake audio files...")
        fake_files = [
            os.path.join(fake_audio_dir, f)
            for f in os.listdir(fake_audio_dir)
            if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))
        ]
        if not real_files:
            raise ValueError(f"No audio files found in {real_audio_dir}")
        if not fake_files:
            raise ValueError(f"No audio files found in {fake_audio_dir}")

        all_files = real_files + fake_files
        all_labels = [1] * len(real_files) + [0] * len(fake_files)

        print(f"Total: {len(all_files)} files  (Real=1: {len(real_files)}, Fake=0: {len(fake_files)})")
        print("Generating spectrograms — this may take a while...")

        specs, labels = [], []
        for path, lbl in zip(all_files, all_labels):
            mel = self.spec_generator.generate_melspectrogram(
                path, duration=self.clip_duration
            )
            if mel is not None:
                specs.append(self.spec_generator.prepare_for_cnn(mel, target_shape))
                labels.append(lbl)

        if not specs:
            raise ValueError("No spectrograms generated — check audio files.")

        X = np.array(specs, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)

        unique, counts = np.unique(y, return_counts=True)
        print(f"Spectrogram array: {X.shape},  labels: {y.shape}")
        for cls, cnt in zip(unique, counts):
            print(f"  class {cls} ({'REAL' if cls == 1 else 'FAKE'}): {cnt} spectrograms")

        return X, y

    def train(
        self,
        X,
        y,
        test_size=0.15,
        val_size=0.20,
        epochs=100,
        batch_size=32,
        augment_factor=1,
    ):
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        X_train_clean, X_val, y_train_clean, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=val_size / (1 - test_size),
            random_state=42,
            stratify=y_trainval,
        )

        print(f"\nSplit sizes:")
        print(f"  Train (before aug): {len(X_train_clean)}")
        print(f"  Validation (clean): {len(X_val)}")
        print(f"  Test  (held-out):   {len(X_test)}")

        print(f"\nAugmenting training set (×{augment_factor + 1})...")
        X_train, y_train_aug = augment_dataset(X_train_clean, y_train_clean, factor=augment_factor)

        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm]
        y_train_aug = y_train_aug[perm]
        print(f"  Train (after aug):  {len(X_train)}")

        unique, counts = np.unique(y_train_aug, return_counts=True)
        for cls, cnt in zip(unique, counts):
            print(f"  Aug label {cls} ({'REAL' if cls==1 else 'FAKE'}): {cnt}")

        cw_values = compute_class_weight(
            class_weight='balanced', classes=np.unique(y_train_aug), y=y_train_aug
        )
        class_weight_dict = {int(k): float(v) for k, v in zip(np.unique(y_train_aug), cw_values)}
        print(f"\nClass weights: {class_weight_dict}")

        self.model = build_cnn_model(self.input_shape)
        print("\nModel summary:")
        self.model.summary()

        os.makedirs('models/dl_model', exist_ok=True)

        callbacks = [
            EarlyStopping(
                monitor='val_auc',
                patience=10,
                mode='max',
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                'models/dl_model/best_model.keras',
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        print("\nTraining CNN model...")
        history = self.model.fit(
            X_train,
            y_train_aug,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1,
        )

        print("\n" + "=" * 55)
        print("EVALUATION ON HELD-OUT TEST SET")
        print("=" * 55)
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metric_names = ['loss', 'accuracy', 'auc', 'precision', 'recall']
        for name, val in zip(metric_names, results):
            print(f"  {name:12s}: {val:.4f}")

        y_prob = self.model.predict(X_test, verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)

        print(f"\np_real on test — min={y_prob.min():.3f}  max={y_prob.max():.3f}  mean={y_prob.mean():.3f}")
        if y_prob.max() < 0.6:
            print("🚨 WARNING: Model never predicts REAL with high confidence — possible label inversion!")
        if y_prob.min() > 0.4:
            print("🚨 WARNING: Model never predicts FAKE — biased toward REAL!")

        unique_preds = np.unique(y_pred)
        if len(unique_preds) == 1:
            print(f"\n🚨 WARNING: model predicts ONLY class {unique_preds[0]} on test set!")
        else:
            print("\n✓ Model predicts both classes on the test set.")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake(0)', 'Real(1)']))
        print("Confusion Matrix (rows=true, cols=pred):")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        print(f"  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}")

        best_epoch = np.argmax(history.history['val_auc'])
        train_acc = history.history['accuracy'][best_epoch]
        val_acc = history.history['val_accuracy'][best_epoch]
        gap = train_acc - val_acc
        print(
            f"\nOverfit check (best epoch {best_epoch+1}) — "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  gap={gap:.4f}"
        )
        if gap > 0.08:
            print("⚠ Possible overfitting (gap > 0.08). Try more data or stronger dropout.")
        else:
            print("✓ Train/val accuracy gap is acceptable.")

        return history, {
            'accuracy': results[1],
            'auc': results[2],
            'y_test': y_test,
            'y_pred': y_pred,
        }

    def save_model(self, model_dir='models/dl_model'):
        os.makedirs(model_dir, exist_ok=True)
        if self.model is None:
            raise ValueError("Model is None — train first.")

        keras_path = os.path.join(model_dir, 'cnn_model.keras')
        print(f"\nSaving model → {keras_path}")
        self.model.save(keras_path)

        size_mb = os.path.getsize(keras_path) / 1024 / 1024
        print(f"  File size: {size_mb:.2f} MB")
        if size_mb < 1:
            raise RuntimeError("Saved model is suspiciously small — likely corrupt.")

        h5_path = os.path.join(model_dir, 'cnn_model.h5')
        try:
            self.model.save(h5_path)
            print(f"  Also saved as: {h5_path}")
        except Exception as e:
            print(f"  Could not save .h5 (OK): {e}")

        print("Validating saved model...")
        loaded = keras.models.load_model(keras_path, compile=False)
        test_in = np.random.rand(1, *self.input_shape).astype(np.float32)
        test_out = float(loaded.predict(test_in, verbose=0)[0][0])
        print(f"  Round-trip prediction on noise: {test_out:.4f}")
        print("✓ Model saved and validated successfully.")


def main():
    print("=" * 55)
    print("DEEP LEARNING MODEL TRAINING (FULLY FIXED)")
    print("=" * 55)
    print(f"Clip duration for training: {CLIP_DURATION}s")
    print("This MUST match _DL_CLIP_DURATION in utils/hybrid_predictor.py")

    trainer = CNNModelTrainer(input_shape=(128, 128, 1))
    real_dir = 'data/train/real'
    fake_dir = 'data/train/fake'

    for d in (real_dir, fake_dir):
        if not os.path.exists(d):
            print(f"\n❌ Directory not found: {d}")
            return

    try:
        X, y = trainer.load_data(real_dir, fake_dir)
        history, metrics = trainer.train(
            X,
            y,
            epochs=100,
            batch_size=32,
            augment_factor=1,
        )
        trainer.save_model()

        print("\n" + "=" * 55)
        print("✅ TRAINING COMPLETED")
        print(f"   Test accuracy : {metrics['accuracy']:.4f}")
        print(f"   Test AUC      : {metrics['auc']:.4f}")
        print("=" * 55)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()