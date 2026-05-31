# config.py
"""
Configuration for Power System Fault Prediction
Tanzania 230V/400V Distribution Network
CNN-Wavelet Deep Learning Model
"""

import os
from pathlib import Path

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

for dir_path in [DATA_DIR, RESULTS_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
DATA_FILE = "TZ_PowerFault_LSTM_Dataset.csv"
DATA_PATH = DATA_DIR / DATA_FILE

# Column names (must match dataset exactly)
SEQUENCE_ID_COL = "Sequence_ID"
TIMESTEP_COL = "Timestep"
VOLTAGE_COL = "Voltage_V"
CURRENT_COL = "Current_A"
TEMPERATURE_COL = "Temperature_C"
FAULT_TYPE_COL = "Fault_Type"

# ============================================================================
# FAULT TYPES (matches your dataset)
# ============================================================================
FAULT_TYPES = [
    "Normal",
    "Overload",
    "Transformer_Overheating",
    "Loose_Connection",
]

# ============================================================================
# WINDOW CONFIGURATION
# ============================================================================
WINDOW_SIZE = 10
STRIDE = 20

# ============================================================================
# TANZANIA GRID PARAMETERS
# ============================================================================
NOMINAL_VOLTAGE = 230.0
VOLTAGE_TOLERANCE = 0.10
NOMINAL_FREQUENCY = 50
AMBIENT_TEMP_MIN = 25.0
AMBIENT_TEMP_MAX = 45.0

# ============================================================================
# NOISE MODEL PARAMETERS
# ============================================================================
NOISE_CONFIG = {
    "temporal_correlation": 0.7,
    "temporal_std": 0.02,
    "load_noise_factor": 0.05,
    "current_threshold": 15.0,
    "switching_probability": 0.08,
    "switching_magnitude_range": (5, 20),
    "recovery_time": 5,
    "temp_voltage_coeff": -0.5,
    "humidity_factor_range": (0.98, 1.02),
    "voltage_sensor_accuracy": 0.02,
    "current_sensor_accuracy": 0.03,
    "temp_sensor_accuracy": 1.0,
    "spike_probability": 0.02,
    "spike_magnitude_factor": 3.0,
}

# ============================================================================
# WAVELET CONFIGURATION
# ============================================================================
WAVELET_CONFIG = {
    "wavelet_type": "db4",
    "decomposition_level": 4,
    "threshold_method": "soft",
    "threshold_mode": "universal",
    "detail_coeffs_to_denoise": [1, 2, 3],
}

# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================
FEATURE_CONFIG = {
    "rolling_mean_window": 5,
    "rolling_var_window": 5,
    "thermal_accumulation_decay": 0.95,
    "thermal_warning_threshold": 45.0,
    "voltage_fluctuation_window": 5,
    "current_instability_window": 5,
    "anomaly_sensitivity": 2.5,
    "use_electrical_features": True,
    "use_thermal_features": True,
    "use_stability_features": True,
    "use_wavelet_features": True,
}

# ============================================================================
# MODEL ARCHITECTURE CONFIGURATION
# ============================================================================
MODEL_CONFIG = {
    "model_type": "cnn",
    "cnn_filters": [128, 256, 512],
    "cnn_kernel_sizes": [3, 5, 3],
    "cnn_activation": "relu",
    "cnn_padding": "same",
    "dropout_rate": 0.2,
    "batch_normalization": True,
    "l2_reg": 1e-5,
    "global_pool_type": "max",
    "lstm_units": [64],
    "lstm_dropout": 0.2,
    "lstm_recurrent_dropout": 0.1,
    "use_lstm_only_for_trends": True,
    "dense_units": [256, 128],
    "dense_activation": "relu",
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
TRAINING_CONFIG = {
    "test_size": 0.10,
    "validation_size": 0.10,
    "random_state": 42,
    "batch_size": 128,
    "epochs": 100,
    "learning_rate": 0.001,
    "use_class_weights": False,  # Dataset is perfectly balanced
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.001,
    "early_stopping_restore_best": True,
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.5,
    "reduce_lr_min_lr": 1e-6,
    "validation_freq": 1,
    "save_best_only": True,
    "model_filename": "fault_prediction_model.h5",
}

# ============================================================================
# PREDICTION CONFIGURATION
# ============================================================================
PREDICTION_CONFIG = {
    "confidence_threshold": 0.6,
    "top_k": 3,
    "buffer_max_size": 50,
}

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
VISUALIZATION_CONFIG = {
    "figure_dpi": 150,
    "figure_format": "png",
    "confusion_matrix_cmap": "Blues",
    "plot_style": "seaborn-v0_8-darkgrid",
}

# ============================================================================
# PHYSICS CHECKS
# ============================================================================
PHYSICS_CHECKS = {
    "voltage_absolute_min": 0,
    "voltage_absolute_max": 300,
    "current_absolute_min": 0,
    "current_absolute_max": 100,
    "temperature_absolute_min": 0,
    "temperature_absolute_max": 150,
}
