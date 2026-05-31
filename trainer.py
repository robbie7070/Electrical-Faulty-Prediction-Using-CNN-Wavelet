# trainer.py
"""
CNN-Wavelet Model Trainer with clean Keras progress display
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import json
import os

# Import from YOUR utils.py - ONLY what exists
from utils import (
    print_section,       # ✅ Exists in your utils
)

# Import config
import config as cfg


# Suppress TensorFlow warnings for clean output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


class Trainer:
    """CNN-Wavelet model trainer with clean progress display."""
    
    def __init__(self, model: keras.Model, config: dict = None):
        """
        Initialize trainer for CNN-Wavelet model.
        
        Parameters
        ----------
        model : keras.Model
            Compiled CNN model
        config : dict
            Training configuration
        """
        self.model = model
        self.config = config or {}
        self.history = None
        
        print("✅ Trainer initialized")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> keras.callbacks.History:
        """
        Train CNN-Wavelet model.
        
        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, window_size, n_features)
        y_train : np.ndarray, shape (n_samples,)
        X_val : np.ndarray
        y_val : np.ndarray
        
        Returns
        -------
        keras.callbacks.History
        """
        
        # Get config
        batch_size = self.config.get('batch_size', 128)
        epochs = self.config.get('epochs', 50)
        use_class_weights = self.config.get('use_class_weights', False)
        patience = self.config.get('early_stopping_patience', 10)
        min_delta = self.config.get('early_stopping_min_delta', 0.001)
        lr_patience = self.config.get('reduce_lr_patience', 5)
        lr_factor = self.config.get('reduce_lr_factor', 0.5)
        min_lr = self.config.get('reduce_lr_min_lr', 1e-6)
        
        # Model save path
        model_path = str(cfg.MODELS_DIR / cfg.TRAINING_CONFIG.get('model_filename', 'cnn_fault_model.h5'))
        
        # Class weights
        class_weight = None
        if use_class_weights:
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight = dict(zip(classes, weights))
            print(f"   Using class weights: {class_weight}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=lr_factor,
                patience=lr_patience,
                min_lr=min_lr,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
        ]
        
        # Print training info using print_section from utils
        print_section("CNN-WAVELET TRAINING")
        print(f"  ├── Architecture: 1D CNN + Wavelet Features")
        print(f"  ├── Training samples: {X_train.shape[0]:,}")
        print(f"  ├── Validation samples: {X_val.shape[0]:,}")
        print(f"  ├── Input shape: {X_train.shape[1:]} (window={X_train.shape[1]}, features={X_train.shape[2]})")
        print(f"  ├── Batch size: {batch_size}")
        print(f"  ├── Max epochs: {epochs}")
        print(f"  ├── Early stopping patience: {patience}")
        print(f"  └── Model save path: {model_path}")
        print()
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )
        
        # Results
        best_epoch = np.argmax(self.history.history['val_accuracy']) + 1
        best_val_acc = max(self.history.history['val_accuracy'])
        best_train_acc = self.history.history['accuracy'][best_epoch - 1]
        
        print_section("CNN-WAVELET TRAINING COMPLETE")
        print(f"  ├── Best epoch: {best_epoch}/{len(self.history.history['val_accuracy'])}")
        print(f"  ├── Train accuracy: {best_train_acc:.4f}")
        print(f"  ├── Val accuracy: {best_val_acc:.4f}")
        print(f"  └── Model saved: {model_path}")
        print()
        
        return self.history
    
    def save_history(self, path: str = None):
        """Save training history."""
        if self.history is None:
            print("⚠️  No training history to save")
            return
        
        if path is None:
            path = str(cfg.RESULTS_DIR / 'training_history.json')
        
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"💾 Training history saved to {path}")