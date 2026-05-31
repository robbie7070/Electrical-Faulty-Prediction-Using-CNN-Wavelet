# preprocessor.py
"""
Preprocessing module with feature engineering, wavelet processing, and scaling
Optimized for large datasets (4M+ records)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import time
import config
from utils import (
    wavelet_denoise,
    extract_wavelet_features,
    apply_realistic_noise,
    rolling_window_statistics,
)


class Preprocessor:
    """Preprocess power system data with feature engineering."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
    
    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================
    
    def engineer_features(self, voltage, current, temperature, timesteps=None):
        """Engineer features from raw electrical measurements."""
        features = {}
        length = len(voltage)
        
        if length < 2:
            return self._engineer_minimal_features(voltage, current, temperature)
        
        if config.FEATURE_CONFIG['use_electrical_features']:
            features.update(self._engineer_electrical(voltage, current, length))
        
        if config.FEATURE_CONFIG['use_thermal_features']:
            features.update(self._engineer_thermal(temperature, timesteps, length))
        
        if config.FEATURE_CONFIG['use_stability_features']:
            features.update(self._engineer_stability(voltage, current, length))
        
        if config.FEATURE_CONFIG['use_wavelet_features']:
            features.update(self._engineer_wavelet(voltage, current, temperature, length))
        
        features_df = pd.DataFrame(features)
        self.feature_names = list(features_df.columns)
        return features_df.values
    
    def _engineer_minimal_features(self, voltage, current, temperature):
        """Minimal features for small arrays."""
        length = len(voltage)
        features = {
            'active_power': voltage * current,
            'voltage_deviation': voltage - config.NOMINAL_VOLTAGE,
            'voltage_ratio': voltage / config.NOMINAL_VOLTAGE,
            'current_ratio': np.ones_like(current),
            'voltage_rolling_mean': np.full_like(voltage, np.mean(voltage)),
            'voltage_rolling_std': np.zeros_like(voltage),
            'current_rolling_mean': np.full_like(current, np.mean(current)),
            'current_rolling_std': np.zeros_like(current),
            'resistance': voltage / (current + 1e-6),
            'dT_dt': np.zeros_like(temperature),
            'thermal_accumulation': np.full_like(temperature, np.mean(temperature)),
            'thermal_stress': np.maximum(0, temperature - config.FEATURE_CONFIG['thermal_warning_threshold']),
            'voltage_fluctuation': np.zeros_like(voltage),
            'current_instability': np.zeros_like(current),
            'anomaly_score': np.zeros_like(voltage),
        }
        features_df = pd.DataFrame(features)
        self.feature_names = list(features_df.columns)
        return features_df.values
    
    def _engineer_electrical(self, voltage, current, length):
        """Electrical features."""
        features = {}
        features['active_power'] = voltage * current
        features['voltage_deviation'] = voltage - config.NOMINAL_VOLTAGE
        features['current_deviation'] = current - np.mean(current)
        features['voltage_ratio'] = voltage / config.NOMINAL_VOLTAGE
        features['current_ratio'] = current / (np.mean(current) + 1e-6)
        
        window = min(config.FEATURE_CONFIG['rolling_mean_window'], length)
        if window >= 2:
            rm, rs, rv = rolling_window_statistics(voltage, window)
            features['voltage_rolling_mean'] = rm
            features['voltage_rolling_std'] = rs
            features['voltage_rolling_var'] = rv
            rm, rs, _ = rolling_window_statistics(current, window)
            features['current_rolling_mean'] = rm
            features['current_rolling_std'] = rs
        else:
            features['voltage_rolling_mean'] = np.full(length, np.mean(voltage))
            features['voltage_rolling_std'] = np.full(length, np.std(voltage))
            features['voltage_rolling_var'] = np.full(length, np.var(voltage))
            features['current_rolling_mean'] = np.full(length, np.mean(current))
            features['current_rolling_std'] = np.full(length, np.std(current))
        
        features['resistance'] = voltage / (current + 1e-6)
        return features
    
    def _engineer_thermal(self, temperature, timesteps, length):
        """Thermal features."""
        features = {}
        if length >= 2:
            if timesteps is not None and len(timesteps) == length and len(timesteps) >= 2:
                features['dT_dt'] = np.gradient(temperature, timesteps)
            else:
                features['dT_dt'] = np.gradient(temperature)
        else:
            features['dT_dt'] = np.zeros(length)
        
        decay = config.FEATURE_CONFIG['thermal_accumulation_decay']
        thermal_accum = np.zeros(length)
        if length > 0:
            thermal_accum[0] = temperature[0]
            for i in range(1, length):
                thermal_accum[i] = decay * thermal_accum[i-1] + (1 - decay) * temperature[i]
        features['thermal_accumulation'] = thermal_accum
        features['thermal_stress'] = np.maximum(0, temperature - config.FEATURE_CONFIG['thermal_warning_threshold'])
        return features
    
    def _engineer_stability(self, voltage, current, length):
        """Stability features - VECTORIZED."""
        features = {}
        window = min(config.FEATURE_CONFIG['voltage_fluctuation_window'], length)
        features['voltage_fluctuation'] = pd.Series(voltage).rolling(
            window=window, min_periods=1).std().fillna(0).values
        features['voltage_fluctuation_normalized'] = features['voltage_fluctuation'] / (config.NOMINAL_VOLTAGE + 1e-6)
        
        window = min(config.FEATURE_CONFIG['current_instability_window'], length)
        cs = pd.Series(current).rolling(window=window, min_periods=1).std().fillna(0).values
        cm = pd.Series(current).rolling(window=window, min_periods=1).mean().fillna(1).values
        features['current_instability'] = np.where(cm > 1e-6, cs / (cm + 1e-6), 0)
        
        vm, vs = np.mean(voltage), np.std(voltage)
        if vs > 1e-6:
            features['anomaly_score'] = (np.abs((voltage - vm) / vs) > config.FEATURE_CONFIG['anomaly_sensitivity']).astype(float)
        else:
            features['anomaly_score'] = np.zeros(length)
        return features
    
    def _engineer_wavelet(self, voltage, current, temperature, length):
        """Wavelet features."""
        features = {}
        max_lvl = int(np.log2(max(length, 4))) - 1 if length >= 4 else 1
        lvl = min(config.WAVELET_CONFIG['decomposition_level'], max(max_lvl, 1))
        wt = config.WAVELET_CONFIG['wavelet_type']
        
        if len(voltage) >= 4:
            wf = extract_wavelet_features(voltage, wt, lvl)
            for k, v in wf.items(): features[f'wav_v_{k}'] = v
        
        if len(current) >= 4:
            wf = extract_wavelet_features(current, wt, lvl)
            for k, v in wf.items(): features[f'wav_c_{k}'] = v
        
        if len(temperature) >= 4:
            wf = extract_wavelet_features(temperature, wt, lvl)
            for k, v in wf.items(): features[f'wav_t_{k}'] = v
        return features
    
    # =========================================================================
    # SEQUENCE CREATION - OPTIMIZED
    # =========================================================================
    
    def create_sequences(self, data, labels, sequence_ids, timesteps):
        """Create sliding window sequences using pandas groupby (FAST)."""
        start_time = time.time()
        
        # Create DataFrame for fast groupby
        df = pd.DataFrame({
            'seq_id': sequence_ids,
            'label': labels,
            'data_idx': np.arange(len(labels))
        })
        
        print(f"   Grouping {len(df):,} records...")
        grouped = df.groupby('seq_id')
        total_seqs = len(grouped)
        
        # Estimate windows
        seq_lengths = grouped.size()
        valid_lengths = seq_lengths[seq_lengths >= config.WINDOW_SIZE]
        estimated = int(((valid_lengths - config.WINDOW_SIZE) // config.STRIDE + 1).sum())
        
        print(f"   Sequences: {total_seqs:,}")
        print(f"   Valid (>= {config.WINDOW_SIZE}): {len(valid_lengths):,}")
        print(f"   Estimated windows: {estimated:,}")
        print(f"   Creating windows...")
        
        X_sequences = []
        y_sequences = []
        total_windows = 0
        last_print = 0
        
        for seq_id, group in grouped:
            seq_len = len(group)
            
            if seq_len < config.WINDOW_SIZE:
                continue
            
            indices = group['data_idx'].values
            
            for j in range(0, seq_len - config.WINDOW_SIZE + 1, config.STRIDE):
                window_indices = indices[j:j + config.WINDOW_SIZE]
                X_sequences.append(data[window_indices])
                y_sequences.append(labels[indices[j + config.WINDOW_SIZE - 1]])
                total_windows += 1
            
            # Progress every 3 seconds
            if time.time() - last_print > 3 and estimated > 0:
                pct = total_windows / estimated * 100
                elapsed = time.time() - start_time
                rate = total_windows / elapsed if elapsed > 0 else 0
                eta = (estimated - total_windows) / rate if rate > 0 else 0
                print(f"   {total_windows:,}/{estimated:,} ({pct:.0f}%) | "
                      f"{rate:.0f} w/s | ETA: {eta:.0f}s")
                last_print = time.time()
        
        elapsed = time.time() - start_time
        print(f"   Complete: {total_windows:,} windows in {elapsed:.1f}s "
              f"({total_windows/elapsed:.0f} w/s)")
        
        return np.array(X_sequences), np.array(y_sequences)
    
    # =========================================================================
    # NOISE
    # =========================================================================
    
    def add_realistic_noise(self, voltage, current, temperature, seed=None):
        """Add realistic field noise."""
        return apply_realistic_noise(voltage, current, temperature, config.NOISE_CONFIG, seed)
    
    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================
    
    def fit_transform(self, df, sequences):
        """Full preprocessing pipeline."""
        print("\n" + "="*60)
        print("🔧 PREPROCESSING DATA")
        print("="*60)
        
        # Extract data
        voltage = df[config.VOLTAGE_COL].values.astype(np.float64)
        current = df[config.CURRENT_COL].values.astype(np.float64)
        temperature = df[config.TEMPERATURE_COL].values.astype(np.float64)
        sequence_ids = df[config.SEQUENCE_ID_COL].values
        timesteps = df[config.TIMESTEP_COL].values
        labels = df[config.FAULT_TYPE_COL].values
        
        if len(voltage) == 0:
            raise ValueError("No data to preprocess!")
        
        print(f"   Records: {len(voltage):,}")
        print(f"   Voltage: {voltage.min():.1f} - {voltage.max():.1f}V")
        print(f"   Current: {current.min():.1f} - {current.max():.1f}A")
        print(f"   Temp: {temperature.min():.1f} - {temperature.max():.1f}°C")
        
        # Add noise
        print("\n⚡ Adding realistic field noise...")
        voltage_noisy, current_noisy, temp_noisy = self.add_realistic_noise(voltage, current, temperature)
        
        # Wavelet denoising
        print("🌊 Applying wavelet processing...")
        voltage_denoised = wavelet_denoise(
            voltage_noisy,
            wavelet=config.WAVELET_CONFIG['wavelet_type'],
            level=min(config.WAVELET_CONFIG['decomposition_level'],
                      int(np.log2(max(len(voltage_noisy), 4))) - 1),
            threshold_method=config.WAVELET_CONFIG['threshold_method'],
            threshold_mode=config.WAVELET_CONFIG['threshold_mode'],
            detail_coeffs_to_denoise=config.WAVELET_CONFIG['detail_coeffs_to_denoise']
        )
        
        # Engineer features
        print("🔬 Engineering features...")
        features = self.engineer_features(voltage_denoised, current_noisy, temp_noisy, timesteps)
        
        # Encode labels
        print("🏷️  Encoding labels...")
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        n_classes = len(self.label_encoder.classes_)
        print(f"   Classes: {n_classes}")
        for i, cls in enumerate(self.label_encoder.classes_):
            count = np.sum(labels == cls)
            print(f"      {i}: {cls} ({count:,})")
        
        # Scale features
        print("📏 Scaling features...")
        features_scaled = self.scaler.fit_transform(features)
        
        # Save artifacts
        joblib.dump(self.scaler, str(config.MODELS_DIR / 'scaler.pkl'))
        joblib.dump(self.label_encoder, str(config.MODELS_DIR / 'encoder.pkl'))
        
        # Create sequences
        print(f"\n📦 Creating sequences (window={config.WINDOW_SIZE}, stride={config.STRIDE})...")
        X, y = self.create_sequences(features_scaled, labels_encoded, sequence_ids, timesteps)
        
        print(f"\n📊 Final: {X.shape[0]:,} windows, shape={X.shape}")
        
        # Split data
        print(f"\n✂️  Splitting data...")
        test_size = config.TRAINING_CONFIG['test_size']
        val_ratio = config.TRAINING_CONFIG['validation_size'] / (1 - test_size)
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size,
            random_state=config.TRAINING_CONFIG['random_state'],
            stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio,
            random_state=config.TRAINING_CONFIG['random_state'],
            stratify=y_temp
        )
        
        print(f"   Train: {X_train.shape[0]:,}")
        print(f"   Validation: {X_val.shape[0]:,}")
        print(f"   Test: {X_test.shape[0]:,}")
        print("="*60 + "\n")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, self.label_encoder
    
    def transform(self, voltage, current, temperature, timesteps=None):
        """Transform new data for prediction."""
        voltage_denoised = wavelet_denoise(
            voltage,
            wavelet=config.WAVELET_CONFIG['wavelet_type'],
            level=min(config.WAVELET_CONFIG['decomposition_level'],
                      int(np.log2(max(len(voltage), 4))) - 1),
            threshold_method=config.WAVELET_CONFIG['threshold_method'],
            threshold_mode=config.WAVELET_CONFIG['threshold_mode'],
            detail_coeffs_to_denoise=config.WAVELET_CONFIG['detail_coeffs_to_denoise']
        )
        features = self.engineer_features(voltage_denoised, current, temperature, timesteps)
        return self.scaler.transform(features)