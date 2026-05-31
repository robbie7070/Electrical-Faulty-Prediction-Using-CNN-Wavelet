# evaluator.py
"""
Model evaluation with metrics, confusion matrix, and classification report
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
import config


class Evaluator:
    """Model evaluator for fault prediction."""
    
    def __init__(self, model, label_encoder):
        """
        Initialize evaluator.
        
        Parameters:
        -----------
        model : keras.Model
            Trained model
        label_encoder : LabelEncoder
            Fitted label encoder
        """
        self.model = model
        self.label_encoder = label_encoder
        self.results = {}
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
            
        Returns:
        --------
        dict : Evaluation results
        """
        print("\n" + "="*60)
        print("📊 EVALUATING MODEL")
        print("="*60)
        
        # Predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, per_class_support = \
            precision_recall_fscore_support(y_test, y_pred, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_names = self.label_encoder.classes_
        report = classification_report(
            y_test, y_pred,
            target_names=class_names,
            digits=3
        )
        
        # Store results
        self.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'per_class_metrics': {
                name: {
                    'precision': p,
                    'recall': r,
                    'f1_score': f,
                    'support': s
                }
                for name, p, r, f, s in zip(
                    class_names,
                    per_class_precision,
                    per_class_recall,
                    per_class_f1,
                    per_class_support
                )
            }
        }
        
        # Print results
        print(f"\n📈 Overall Metrics:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        
        print(f"\n📋 Classification Report:")
        print(report)
        
        print(f"\n🗂️  Per-Class Breakdown:")
        for cls, metrics in self.results['per_class_metrics'].items():
            print(f"   {cls}:")
            print(f"      Precision: {metrics['precision']:.3f}")
            print(f"      Recall:    {metrics['recall']:.3f}")
            print(f"      F1-Score:  {metrics['f1_score']:.3f}")
            print(f"      Support:   {metrics['support']}")
        
        print("="*60 + "\n")
        
        return self.results
    
    def save_results(self):
        """Save evaluation results to CSV."""
        if not self.results:
            print("No results to save. Run evaluate() first.")
            return
        
        # Per-class metrics
        metrics_df = pd.DataFrame(self.results['per_class_metrics']).T
        metrics_df.to_csv(config.RESULTS_DIR / 'per_class_metrics.csv')
        
        # Confusion matrix
        cm_df = pd.DataFrame(
            self.results['confusion_matrix'],
            index=self.label_encoder.classes_,
            columns=self.label_encoder.classes_
        )
        cm_df.to_csv(config.RESULTS_DIR / 'confusion_matrix.csv')
        
        print(f"📝 Results saved to {config.RESULTS_DIR}")
        
    def get_misclassified_samples(self, X_test, y_test, max_samples=10):
        """
        Get misclassified samples for analysis.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        max_samples : int
            Maximum samples to return
            
        Returns:
        --------
        pd.DataFrame : Misclassified samples
        """
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        misclassified_mask = y_pred != y_test
        misclassified_indices = np.where(misclassified_mask)[0]
        
        if len(misclassified_indices) > max_samples:
            misclassified_indices = misclassified_indices[:max_samples]
        
        results = []
        for idx in misclassified_indices:
            true_label = self.label_encoder.inverse_transform([y_test[idx]])[0]
            pred_label = self.label_encoder.inverse_transform([y_pred[idx]])[0]
            confidence = np.max(y_pred_proba[idx])
            
            results.append({
                'sample_index': idx,
                'true_fault': true_label,
                'predicted_fault': pred_label,
                'confidence': confidence,
            })
        
        return pd.DataFrame(results)