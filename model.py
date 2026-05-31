# model.py
"""
CNN-based model architecture with optional LSTM for temporal refinement
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import config


class FaultPredictionModel:
    """CNN-based model for power system fault prediction."""
    
    def __init__(self, input_shape, num_classes):
        """
        Initialize model.
        
        Parameters:
        -----------
        input_shape : tuple
            (window_size, num_features)
        num_classes : int
            Number of fault classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_cnn_model(self):
        """Build primary 1D CNN model."""
        
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        x = inputs
        
        # CNN blocks
        for i, (filters, kernel_size) in enumerate(zip(
            config.MODEL_CONFIG['cnn_filters'],
            config.MODEL_CONFIG['cnn_kernel_sizes']
        )):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding=config.MODEL_CONFIG['cnn_padding'],
                activation=None,
                kernel_regularizer=regularizers.l2(config.MODEL_CONFIG['l2_reg']),
                name=f'conv1d_{i+1}'
            )(x)
            
            if config.MODEL_CONFIG['batch_normalization']:
                x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
            
            x = layers.Activation(
                config.MODEL_CONFIG['cnn_activation'],
                name=f'activation_{i+1}'
            )(x)
            
            x = layers.Dropout(
                config.MODEL_CONFIG['dropout_rate'],
                name=f'dropout_{i+1}'
            )(x)
        
        # Global pooling
        if config.MODEL_CONFIG['global_pool_type'] == 'max':
            x = layers.GlobalMaxPooling1D(name='global_pool')(x)
        else:
            x = layers.GlobalAveragePooling1D(name='global_pool')(x)
        
        # Dense layers
        for i, units in enumerate(config.MODEL_CONFIG['dense_units']):
            x = layers.Dense(
                units=units,
                activation=None,
                kernel_regularizer=regularizers.l2(config.MODEL_CONFIG['l2_reg']),
                name=f'dense_{i+1}'
            )(x)
            
            if config.MODEL_CONFIG['batch_normalization']:
                x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
            
            x = layers.Activation(
                config.MODEL_CONFIG['dense_activation'],
                name=f'activation_dense_{i+1}'
            )(x)
            
            x = layers.Dropout(
                config.MODEL_CONFIG['dropout_rate'] * 0.5,
                name=f'dropout_dense_{i+1}'
            )(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='output'
        )(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='FaultPrediction_CNN')
        
        return self.model
    
    def build_cnn_lstm_model(self):
        """Build CNN-LSTM hybrid model with optional LSTM for trends."""
        
        inputs = layers.Input(shape=self.input_shape, name='input')
        
        # CNN branch (primary feature extraction)
        cnn_branch = inputs
        
        for i, (filters, kernel_size) in enumerate(zip(
            config.MODEL_CONFIG['cnn_filters'],
            config.MODEL_CONFIG['cnn_kernel_sizes']
        )):
            cnn_branch = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding=config.MODEL_CONFIG['cnn_padding'],
                activation=None,
                kernel_regularizer=regularizers.l2(config.MODEL_CONFIG['l2_reg']),
                name=f'conv1d_{i+1}'
            )(cnn_branch)
            
            if config.MODEL_CONFIG['batch_normalization']:
                cnn_branch = layers.BatchNormalization(name=f'bn_{i+1}')(cnn_branch)
            
            cnn_branch = layers.Activation(
                config.MODEL_CONFIG['cnn_activation'],
                name=f'activation_{i+1}'
            )(cnn_branch)
            
            cnn_branch = layers.Dropout(
                config.MODEL_CONFIG['dropout_rate'],
                name=f'dropout_{i+1}'
            )(cnn_branch)
        
        # LSTM branch (temporal refinement)
        if config.MODEL_CONFIG.get('use_lstm_only_for_trends', True):
            # Use only thermal-related features for LSTM
            # Assumes thermal features are at specific indices (adjust as needed)
            lstm_branch = layers.Lambda(
                lambda x: x[:, :, -10:],  # Last 10 features (thermal and stability)
                name='thermal_features_extractor'
            )(inputs)
        else:
            lstm_branch = inputs
        
        for i, units in enumerate(config.MODEL_CONFIG['lstm_units']):
            return_seq = i < len(config.MODEL_CONFIG['lstm_units']) - 1
            lstm_branch = layers.LSTM(
                units=units,
                return_sequences=return_seq,
                dropout=config.MODEL_CONFIG['lstm_dropout'],
                recurrent_dropout=config.MODEL_CONFIG['lstm_recurrent_dropout'],
                name=f'lstm_{i+1}'
            )(lstm_branch)
        
        # Merge branches
        cnn_pooled = layers.GlobalMaxPooling1D(name='cnn_pool')(cnn_branch)
        
        if len(config.MODEL_CONFIG['lstm_units']) > 0:
            merged = layers.Concatenate(name='merge')([cnn_pooled, lstm_branch])
        else:
            merged = cnn_pooled
        
        # Dense layers
        x = merged
        for i, units in enumerate(config.MODEL_CONFIG['dense_units']):
            x = layers.Dense(
                units=units,
                activation=None,
                kernel_regularizer=regularizers.l2(config.MODEL_CONFIG['l2_reg']),
                name=f'dense_{i+1}'
            )(x)
            
            if config.MODEL_CONFIG['batch_normalization']:
                x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
            
            x = layers.Activation(
                config.MODEL_CONFIG['dense_activation'],
                name=f'activation_dense_{i+1}'
            )(x)
            
            x = layers.Dropout(
                config.MODEL_CONFIG['dropout_rate'] * 0.5,
                name=f'dropout_dense_{i+1}'
            )(x)
        
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='output'
        )(x)
        
        self.model = keras.Model(
            inputs=inputs, 
            outputs=outputs, 
            name='FaultPrediction_CNN_LSTM'
        )
        
        return self.model
    
    def build(self):
        """Build model based on configuration."""
        if config.MODEL_CONFIG['model_type'] == 'cnn':
            print("🏗️  Building CNN model...")
            self.build_cnn_model()
        elif config.MODEL_CONFIG['model_type'] == 'cnn_lstm':
            print("🏗️  Building CNN-LSTM hybrid model...")
            self.build_cnn_lstm_model()
        else:
            raise ValueError(f"Unknown model type: {config.MODEL_CONFIG['model_type']}")
        
        return self.model
    
    def compile_model(self):
        """Compile the model."""
        optimizer = keras.optimizers.Adam(
            learning_rate=config.TRAINING_CONFIG['learning_rate']
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model compiled")
        self.model.summary()
        
        return self.model
    
    def save(self, filepath=None):
        """Save model to disk."""
        if filepath is None:
            filepath = config.MODELS_DIR / config.TRAINING_CONFIG['model_filename']
        
        self.model.save(filepath)
        print(f"💾 Model saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load model from disk."""
        return keras.models.load_model(filepath)