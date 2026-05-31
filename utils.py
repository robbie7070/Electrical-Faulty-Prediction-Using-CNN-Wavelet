# utils.py
"""
Utility functions for signal processing, wavelet transforms, and helpers
Optimized for large datasets
"""

import numpy as np
import pandas as pd
import pywt
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# NOISE FUNCTIONS
# ============================================================================

def add_correlated_noise(signal_data, correlation=0.7, std=0.02, seed=None):
    """Add correlated temporal noise (AR(1) process)."""
    if len(signal_data) == 0:
        return signal_data
    
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.zeros_like(signal_data)
    noise[0] = np.random.normal(0, std)
    
    for i in range(1, len(signal_data)):
        noise[i] = correlation * noise[i-1] + np.random.normal(0, std * np.sqrt(1 - correlation**2))
    
    return signal_data + noise


def add_load_dependent_noise(voltage, current, base_noise_factor=0.05, threshold=15.0):
    """Add noise proportional to current magnitude (load-dependent)."""
    if len(voltage) == 0 or len(current) == 0:
        return voltage
    
    noise_mask = current > threshold
    noise_factor = np.ones_like(voltage)
    noise_factor[noise_mask] += base_noise_factor * (current[noise_mask] - threshold) / (threshold + 1e-6)
    
    noise = np.random.normal(0, 1, len(voltage)) * noise_factor * voltage * 0.01
    return voltage + noise


def add_switching_disturbances(voltage, current, prob=0.08, magnitude_range=(5, 20), recovery_time=5):
    """Add motor/load switching disturbances."""
    if len(voltage) == 0 or len(current) == 0:
        return voltage, current
    
    if np.random.random() > prob:
        return voltage, current
    
    n = len(voltage)
    switch_point = np.random.randint(n // 4, 3 * n // 4)
    dip_magnitude = np.random.uniform(magnitude_range[0], magnitude_range[1])
    
    voltage_disturbed = voltage.copy()
    voltage_disturbed[switch_point:] -= dip_magnitude
    
    recovery = np.exp(-np.arange(n - switch_point) / recovery_time)
    voltage_disturbed[switch_point:] += dip_magnitude * (1 - recovery)
    
    current_disturbed = current.copy()
    current_spike = np.random.uniform(1.5, 3.0) * current[switch_point]
    end_spike = min(switch_point + 3, n)
    current_disturbed[switch_point:end_spike] = current_spike
    
    return voltage_disturbed, current_disturbed


def add_environmental_coupling(voltage, temperature, temp_coeff=-0.5, humidity_range=(0.98, 1.02), seed=None):
    """Add environmental coupling effects (temperature and humidity)."""
    if len(voltage) == 0 or len(temperature) == 0:
        return voltage
    
    if seed is not None:
        np.random.seed(seed)
    
    temp_effect = np.where(temperature > 30, (temperature - 30) * temp_coeff, 0)
    humidity_factor = np.random.uniform(humidity_range[0], humidity_range[1], len(voltage))
    
    return voltage + temp_effect * humidity_factor


def add_sensor_noise(signal_data, accuracy=0.02, spike_prob=0.02, spike_factor=3.0, seed=None):
    """Add sensor measurement noise and occasional spikes."""
    if len(signal_data) == 0:
        return signal_data
    
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.normal(0, accuracy * np.abs(signal_data) + 1e-6, len(signal_data))
    noisy_signal = signal_data + noise
    
    spike_mask = np.random.random(len(signal_data)) < spike_prob
    if spike_mask.any():
        spike_direction = np.random.choice([-1, 1], size=np.sum(spike_mask))
        noisy_signal[spike_mask] *= spike_factor * spike_direction
    
    return noisy_signal


def apply_realistic_noise(voltage, current, temperature, noise_config, seed=None):
    """Apply complete realistic noise model for Tanzania LV system."""
    if len(voltage) == 0 or len(current) == 0 or len(temperature) == 0:
        return voltage, current, temperature
    
    if seed is not None:
        np.random.seed(seed)
    
    # 1. Correlated temporal noise
    voltage = add_correlated_noise(
        voltage,
        noise_config['temporal_correlation'],
        noise_config['temporal_std'],
        seed=seed
    )
    
    # 2. Load-dependent noise
    voltage = add_load_dependent_noise(
        voltage, current,
        noise_config['load_noise_factor'],
        noise_config['current_threshold']
    )
    
    # 3. Switching disturbances
    voltage, current = add_switching_disturbances(
        voltage, current,
        noise_config['switching_probability'],
        noise_config['switching_magnitude_range'],
        noise_config['recovery_time']
    )
    
    # 4. Environmental coupling
    voltage = add_environmental_coupling(
        voltage, temperature,
        noise_config['temp_voltage_coeff'],
        noise_config['humidity_factor_range'],
        seed=seed
    )
    
    # 5. Sensor measurement noise
    voltage = add_sensor_noise(
        voltage,
        noise_config['voltage_sensor_accuracy'],
        noise_config['spike_probability'],
        noise_config['spike_magnitude_factor'],
        seed=seed
    )
    
    current = add_sensor_noise(
        current,
        noise_config['current_sensor_accuracy'],
        noise_config['spike_probability'],
        noise_config['spike_magnitude_factor'],
        seed=seed+1 if seed else None
    )
    
    temperature = add_sensor_noise(
        temperature,
        noise_config['temp_sensor_accuracy'] / 100,
        noise_config['spike_probability'] * 0.5,
        noise_config['spike_magnitude_factor'] * 0.5,
        seed=seed+2 if seed else None
    )
    
    return voltage, current, temperature


# ============================================================================
# WAVELET FUNCTIONS
# ============================================================================

def wavelet_denoise(signal_data, wavelet='db4', level=4, threshold_mode='universal',
                    threshold_method='soft', detail_coeffs_to_denoise=None):
    """Denoise signal using wavelet transform."""
    if len(signal_data) < 4:
        return signal_data
    
    if detail_coeffs_to_denoise is None:
        detail_coeffs_to_denoise = [1, 2, 3]
    
    # Ensure valid decomposition level
    max_level = pywt.dwt_max_level(len(signal_data), pywt.Wavelet(wavelet))
    level = min(level, max_level)
    
    if level < 1:
        return signal_data
    
    # Decompose
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    approx_coeffs = coeffs[0]
    detail_coeffs_list = list(coeffs[1:])
    
    # Apply thresholding
    for idx in detail_coeffs_to_denoise:
        if 1 <= idx <= len(detail_coeffs_list):
            sigma = np.median(np.abs(detail_coeffs_list[idx-1])) / 0.6745
            
            if threshold_mode == 'universal':
                threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
            else:
                threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
            
            if threshold_method == 'soft':
                detail_coeffs_list[idx-1] = pywt.threshold(detail_coeffs_list[idx-1], threshold, mode='soft')
            else:
                detail_coeffs_list[idx-1] = pywt.threshold(detail_coeffs_list[idx-1], threshold, mode='hard')
    
    # Reconstruct
    coeffs_reconstructed = [approx_coeffs] + detail_coeffs_list
    denoised_signal = pywt.waverec(coeffs_reconstructed, wavelet)
    
    # Match original length
    if len(denoised_signal) > len(signal_data):
        denoised_signal = denoised_signal[:len(signal_data)]
    elif len(denoised_signal) < len(signal_data):
        denoised_signal = np.pad(denoised_signal, (0, len(signal_data) - len(denoised_signal)), 'edge')
    
    return denoised_signal


def extract_wavelet_features(signal_data, wavelet='db4', level=4):
    """Extract features from wavelet decomposition."""
    if len(signal_data) < 4:
        return {}
    
    max_level = pywt.dwt_max_level(len(signal_data), pywt.Wavelet(wavelet))
    level = min(level, max_level)
    
    if level < 1:
        return {}
    
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    
    features = {}
    features['approx_energy'] = np.sum(np.square(coeffs[0])) / len(coeffs[0])
    features['approx_mean'] = np.mean(coeffs[0])
    features['approx_std'] = np.std(coeffs[0])
    
    for i, detail in enumerate(coeffs[1:], 1):
        features[f'detail_{i}_energy'] = np.sum(np.square(detail)) / len(detail)
        features[f'detail_{i}_mean'] = np.mean(np.abs(detail))
        features[f'detail_{i}_std'] = np.std(detail)
        features[f'detail_{i}_max'] = np.max(np.abs(detail))
    
    eps = 1e-6
    features['energy_ratio_1_2'] = features.get('detail_1_energy', 0) / (features.get('detail_2_energy', eps) + eps)
    features['energy_ratio_2_3'] = features.get('detail_2_energy', 0) / (features.get('detail_3_energy', eps) + eps)
    features['approx_detail_ratio'] = features['approx_energy'] / (features.get('detail_1_energy', eps) + eps)
    
    return features


def detect_transients(signal_data, wavelet='db4', threshold_factor=3.0):
    """Detect transients using wavelet detail coefficients."""
    if len(signal_data) < 4:
        return []
    
    coeffs = pywt.wavedec(signal_data, wavelet, level=2)
    detail_coeffs = coeffs[1]
    
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745
    threshold = threshold_factor * sigma
    
    transients = np.where(np.abs(detail_coeffs) > threshold)[0]
    return transients.tolist()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def rolling_window_statistics(data, window_size):
    """Compute rolling window statistics using pandas (fast)."""
    if len(data) == 0:
        return np.array([]), np.array([]), np.array([])
    
    if len(data) < window_size:
        mean = np.full_like(data, np.mean(data))
        std = np.full_like(data, np.std(data))
        var = np.full_like(data, np.var(data))
        return mean, std, var
    
    # Use pandas rolling for speed (C implementation)
    s = pd.Series(data)
    
    rolling_mean = s.rolling(window=window_size, center=True, min_periods=1).mean().values
    rolling_std = s.rolling(window=window_size, center=True, min_periods=1).std().values
    rolling_var = s.rolling(window=window_size, center=True, min_periods=1).var().values
    
    # Fill NaN
    rolling_std = np.nan_to_num(rolling_std, nan=0.0)
    rolling_var = np.nan_to_num(rolling_var, nan=0.0)
    
    return rolling_mean, rolling_std, rolling_var


def physics_consistency_check(voltage, current, temperature, config_dict):
    """Check if values satisfy physics constraints."""
    voltage = np.atleast_1d(voltage)
    current = np.atleast_1d(current)
    temperature = np.atleast_1d(temperature)
    
    checks = [
        np.all(voltage >= config_dict['voltage_absolute_min']),
        np.all(voltage <= config_dict['voltage_absolute_max']),
        np.all(current >= config_dict['current_absolute_min']),
        np.all(current <= config_dict['current_absolute_max']),
        np.all(temperature >= config_dict['temperature_absolute_min']),
        np.all(temperature <= config_dict['temperature_absolute_max']),
    ]
    
    return all(checks)


def print_section(title):
    """Print a formatted section title."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def suppress_tensorflow_warnings():
    """Suppress TensorFlow/GPU warnings."""
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
    except:
        pass