# visualizer.py
"""
Visualization module for training curves, confusion matrices, and wavelet comparisons
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import config
import json


class Visualizer:
    """Generate and save all plots."""
    
    def __init__(self):
        plt.style.use(config.VISUALIZATION_CONFIG['plot_style'])
        self.dpi = config.VISUALIZATION_CONFIG['figure_dpi']
        self.format = config.VISUALIZATION_CONFIG['figure_format']
        
    def plot_training_history(self, history=None):
        """
        Plot training accuracy and loss curves.
        
        Parameters:
        -----------
        history : keras.History, optional
            Training history object
        """
        # Try to load history if not provided
        if history is None:
            try:
                with open(config.RESULTS_DIR / 'training_history.json', 'r') as f:
                    history = json.load(f)
                is_dict = True
            except FileNotFoundError:
                print("No training history found.")
                return
        else:
            is_dict = isinstance(history, dict)
        
        if is_dict:
            accuracy = history['accuracy']
            val_accuracy = history['val_accuracy']
            loss = history['loss']
            val_loss = history['val_loss']
        else:
            accuracy = history.history['accuracy']
            val_accuracy = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        ax1.plot(accuracy, label='Training', linewidth=2)
        ax1.plot(val_accuracy, label='Validation', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Training & Validation Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(loss, label='Training', linewidth=2)
        ax2.plot(val_loss, label='Validation', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training & Validation Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            config.RESULTS_DIR / f'training_history.{self.format}',
            dpi=self.dpi,
            bbox_inches='tight'
        )
        plt.close()
        print(f"📊 Training history plot saved")
        
    def plot_confusion_matrix(self, cm, class_names, normalize=True):
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        cm : np.ndarray
            Confusion matrix
        class_names : list
            Class names
        normalize : bool
            Whether to normalize
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap=config.VISUALIZATION_CONFIG['confusion_matrix_cmap'],
            xticklabels=class_names,
            yticklabels=class_names,
            square=True,
            cbar_kws={"shrink": 0.8},
        )
        
        ax.set_xlabel('Predicted Fault Type')
        ax.set_ylabel('True Fault Type')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(
            config.RESULTS_DIR / f'confusion_matrix.{self.format}',
            dpi=self.dpi,
            bbox_inches='tight'
        )
        plt.close()
        print(f"📊 Confusion matrix plot saved")
        
    def plot_metrics_bar_chart(self, per_class_metrics):
        """
        Plot precision, recall, and F1-score bar chart.
        
        Parameters:
        -----------
        per_class_metrics : dict
            Per-class metrics dictionary
        """
        classes = list(per_class_metrics.keys())
        precision = [m['precision'] for m in per_class_metrics.values()]
        recall = [m['recall'] for m in per_class_metrics.values()]
        f1 = [m['f1_score'] for m in per_class_metrics.values()]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Fault Type')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(
            config.RESULTS_DIR / f'metrics_bar_chart.{self.format}',
            dpi=self.dpi,
            bbox_inches='tight'
        )
        plt.close()
        print(f"📊 Metrics bar chart saved")
        
    def plot_wavelet_comparison(self, raw_signal, denoised_signal, signal_name='Signal'):
        """
        Plot raw vs wavelet-denoised signal comparison.
        
        Parameters:
        -----------
        raw_signal : np.ndarray
            Original noisy signal
        denoised_signal : np.ndarray
            Denoised signal
        signal_name : str
            Signal name for title
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        
        ax.plot(raw_signal, label='Raw (Noisy)', alpha=0.5, linewidth=1)
        ax.plot(denoised_signal, label='Wavelet Denoised', linewidth=2)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Magnitude')
        ax.set_title(f'Wavelet Denoising Comparison - {signal_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            config.RESULTS_DIR / f'wavelet_comparison_{signal_name.lower()}.{self.format}',
            dpi=self.dpi,
            bbox_inches='tight'
        )
        plt.close()
        print(f"📊 Wavelet comparison plot for {signal_name} saved")
        
    def plot_prediction_confidence(self, predictions, true_labels=None):
        """
        Plot prediction confidence distribution.
        
        Parameters:
        -----------
        predictions : np.ndarray
            Prediction probabilities
        true_labels : np.ndarray, optional
            True labels for correctness check
        """
        max_probs = np.max(predictions, axis=1)
        
        if true_labels is not None:
            pred_labels = np.argmax(predictions, axis=1)
            correct = pred_labels == true_labels
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if true_labels is not None:
            ax.hist(max_probs[correct], bins=20, alpha=0.7, label='Correct', color='green')
            ax.hist(max_probs[~correct], bins=20, alpha=0.7, label='Incorrect', color='red')
            ax.legend()
        else:
            ax.hist(max_probs, bins=20, alpha=0.7)
        
        ax.set_xlabel('Prediction Confidence')
        ax.set_ylabel('Count')
        ax.set_title('Prediction Confidence Distribution')
        ax.axvline(x=config.PREDICTION_CONFIG['confidence_threshold'], 
                   color='orange', linestyle='--', label='Threshold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(
            config.RESULTS_DIR / f'confidence_distribution.{self.format}',
            dpi=self.dpi,
            bbox_inches='tight'
        )
        plt.close()
        print(f"📊 Confidence distribution plot saved")