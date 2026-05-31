# main.py
"""
Main training orchestration script
"""

import sys
import numpy as np
import tensorflow as tf
import config
from data_loader import DataLoader
from preprocessor import Preprocessor
from model import FaultPredictionModel
from trainer import Trainer
from evaluator import Evaluator
from visualizer import Visualizer


def set_random_seeds():
    """Set random seeds for reproducibility."""
    np.random.seed(config.TRAINING_CONFIG['random_state'])
    tf.random.set_seed(config.TRAINING_CONFIG['random_state'])


def check_gpu():
    """Check GPU availability."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"🚀 GPU(s) available: {[gpu.name for gpu in gpus]}")
    else:
        print("💻 Running on CPU")


def main():
    """Main training pipeline."""
    print("\n" + "="*60)
    print("⚡ POWER SYSTEM FAULT PREDICTION SYSTEM")
    print("   Tanzania 230V/400V Secondary Distribution Network")
    print("="*60)
    
    # Setup
    set_random_seeds()
    check_gpu()
    
    # Step 1: Load data
    print("\n📂 Step 1: Loading Data")
    loader = DataLoader()
    df, sequences = loader.run_full_validation()
    
    # Step 2: Preprocess data
    print("\n🔧 Step 2: Preprocessing")
    preprocessor = Preprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test, encoder = \
        preprocessor.fit_transform(df, sequences)
    
    # Step 3: Build model
    print("\n🏗️  Step 3: Building Model")
    input_shape = (config.WINDOW_SIZE, X_train.shape[2])
    num_classes = len(encoder.classes_)
    
    model_builder = FaultPredictionModel(input_shape, num_classes)
    model_builder.build()
    model = model_builder.compile_model()
    
    # Step 4: Train model
    print("\n🚀 Step 4: Training")
    trainer = Trainer(model)
    history = trainer.train(X_train, y_train, X_val, y_val)
    trainer.save_history()
    
    # Step 5: Evaluate model
    print("\n📊 Step 5: Evaluating")
    evaluator = Evaluator(model, encoder)
    results = evaluator.evaluate(X_test, y_test)
    evaluator.save_results()
    
    # Step 6: Generate visualizations
    print("\n📈 Step 6: Generating Visualizations")
    visualizer = Visualizer()
    
    # Training history plot
    visualizer.plot_training_history(history)
    
    # Confusion matrix
    visualizer.plot_confusion_matrix(
        results['confusion_matrix'],
        encoder.classes_
    )
    
    # Per-class metrics bar chart
    visualizer.plot_metrics_bar_chart(results['per_class_metrics'])
    
    # Confidence distribution
    visualizer.plot_prediction_confidence(
        results['y_pred_proba'],
        results['y_test']
    )
    
    # Wavelet comparison (sample from test data)
    sample_idx = np.random.randint(0, len(X_test))
    sample_voltage = X_test[sample_idx, :, 0]  # First feature (voltage-based)
    visualizer.plot_wavelet_comparison(
        sample_voltage,
        sample_voltage,  # Already denoised in preprocessing
        "Sample Voltage"
    )
    
    # Save model
    model.save(config.MODELS_DIR / config.TRAINING_CONFIG['model_filename'])
    
    # Save config
    import json
    config_dict = {
        'window_size': config.WINDOW_SIZE,
        'model_type': config.MODEL_CONFIG['model_type'],
        'fault_types': config.FAULT_TYPES,
        'nominal_voltage': config.NOMINAL_VOLTAGE,
    }
    with open(config.MODELS_DIR / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ TRAINING PIPELINE COMPLETE")
    print("="*60)
    print(f"\n📦 Output files:")
    print(f"   Model: {config.MODELS_DIR / config.TRAINING_CONFIG['model_filename']}")
    print(f"   Scaler: {config.MODELS_DIR / 'scaler.pkl'}")
    print(f"   Encoder: {config.MODELS_DIR / 'encoder.pkl'}")
    print(f"   Config: {config.MODELS_DIR / 'config.json'}")
    print(f"   Plots: {config.RESULTS_DIR}/")
    
    print(f"\n🔮 To start prediction system, run:")
    print(f"   python predict_menu.py")
    print()


if __name__ == "__main__":
    main()