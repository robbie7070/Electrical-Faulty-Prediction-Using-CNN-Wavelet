# predictor.py - COMPLETE FIXED VERSION
"""
Prediction engine for batch and manual prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import config
from preprocessor import Preprocessor


class Predictor:
    """Prediction engine for fault detection."""
    
    def __init__(self, model_path=None, scaler_path=None, encoder_path=None):
        # Load model
        if model_path is None:
            model_path = str(config.MODELS_DIR / config.TRAINING_CONFIG['model_filename'])
        self.model = keras.models.load_model(model_path)
        
        # Load scaler
        if scaler_path is None:
            scaler_path = str(config.MODELS_DIR / 'scaler.pkl')
        self.scaler = joblib.load(scaler_path)
        
        # Load encoder
        if encoder_path is None:
            encoder_path = str(config.MODELS_DIR / 'encoder.pkl')
        self.encoder = joblib.load(encoder_path)
        
        # Load feature names (saved during training)
        try:
            self.feature_names = joblib.load(str(config.MODELS_DIR / 'feature_names.pkl'))
            print(f"✅ Loaded {len(self.feature_names)} feature names")
        except:
            self.feature_names = None
            print("⚠️  No feature names file found")
        
        # Create preprocessor
        self.preprocessor = Preprocessor()
        self.preprocessor.scaler = self.scaler
        self.preprocessor.label_encoder = self.encoder
        
        print(f"✅ Predictor ready (expects {self.scaler.n_features_in_} features)")
        
    def predict_single(self, voltage, current, temperature):
        """Predict fault type for single sequence."""
        if len(voltage) != config.WINDOW_SIZE:
            raise ValueError(f"Expected window_size={config.WINDOW_SIZE}, got {len(voltage)}")
        
        # Preprocess to get features
        features = self._preprocess_for_prediction(voltage, current, temperature)
        
        # Reshape for model
        X = features.reshape(1, config.WINDOW_SIZE, -1)
        
        # Predict
        probabilities = self.model.predict(X, verbose=0)[0]
        prediction = np.argmax(probabilities)
        confidence = probabilities[prediction]
        
        # Top-k
        top_k_idx = np.argsort(probabilities)[-config.PREDICTION_CONFIG['top_k']:][::-1]
        top_k_probs = probabilities[top_k_idx]
        top_k_labels = self.encoder.inverse_transform(top_k_idx)
        
        fault_type = self.encoder.inverse_transform([prediction])[0]
        
        result = {
            'fault_type': fault_type,
            'confidence': float(confidence),
            'all_probabilities': {
                label: float(prob) for label, prob in zip(self.encoder.classes_, probabilities)
            },
            'top_3_predictions': [
                {'fault_type': label, 'probability': float(prob)}
                for label, prob in zip(top_k_labels, top_k_probs)
            ],
            'high_confidence': confidence >= config.PREDICTION_CONFIG['confidence_threshold'],
        }
        
        return result
    
    def _preprocess_for_prediction(self, voltage, current, temperature):
        """Preprocess data to match training features EXACTLY."""
        from utils import wavelet_denoise
        
        # 1. Wavelet denoising
        max_level = int(np.log2(max(len(voltage), 4))) - 1
        level = min(config.WAVELET_CONFIG['decomposition_level'], max(max_level, 1))
        
        voltage_denoised = wavelet_denoise(
            voltage,
            wavelet=config.WAVELET_CONFIG['wavelet_type'],
            level=level,
            threshold_method=config.WAVELET_CONFIG['threshold_method'],
            threshold_mode=config.WAVELET_CONFIG['threshold_mode'],
            detail_coeffs_to_denoise=config.WAVELET_CONFIG['detail_coeffs_to_denoise']
        )
        
        # 2. Engineer features
        timesteps = np.arange(len(voltage))
        features = self.preprocessor.engineer_features(
            voltage_denoised, current, temperature, timesteps
        )
        
        # 3. MATCH FEATURES TO TRAINING
        expected = self.scaler.n_features_in_
        actual = features.shape[1]
        
        if actual != expected:
            print(f"   ⚠️  Feature mismatch: got {actual}, need {expected}")
            
            # Convert to DataFrame with feature names
            if self.preprocessor.feature_names and len(self.preprocessor.feature_names) == actual:
                features_df = pd.DataFrame(features, columns=self.preprocessor.feature_names)
                
                # If we have training feature names, use only those
                if self.feature_names is not None:
                    # Add missing columns with zeros
                    for col in self.feature_names:
                        if col not in features_df.columns:
                            features_df[col] = 0.0
                    # Keep only training columns in order
                    features_df = features_df[self.feature_names]
                    features = features_df.values
                    print(f"   ✅ Matched to {features.shape[1]} features")
                else:
                    # Pad with zeros
                    if actual < expected:
                        padding = np.zeros((features.shape[0], expected - actual))
                        features = np.hstack([features, padding])
                        print(f"   ✅ Padded to {features.shape[1]} features")
                    else:
                        features = features[:, :expected]
                        print(f"   ✅ Trimmed to {features.shape[1]} features")
        
        # 4. Scale
        features_scaled = self.scaler.transform(features)
        
        return features_scaled
    
    def predict_from_dataframe(self, df):
        """Predict from DataFrame with required columns."""
        if len(df) < config.WINDOW_SIZE:
            raise ValueError(f"Need at least {config.WINDOW_SIZE} samples, got {len(df)}")
        
        voltage = df[config.VOLTAGE_COL].values
        current = df[config.CURRENT_COL].values
        temperature = df[config.TEMPERATURE_COL].values
        
        results = []
        for i in range(0, len(df) - config.WINDOW_SIZE + 1):
            window_voltage = voltage[i:i + config.WINDOW_SIZE]
            window_current = current[i:i + config.WINDOW_SIZE]
            window_temperature = temperature[i:i + config.WINDOW_SIZE]
            
            result = self.predict_single(window_voltage, window_current, window_temperature)
            result['window_start'] = i
            result['window_end'] = i + config.WINDOW_SIZE
            results.append(result)
        
        return results