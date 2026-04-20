#!/usr/bin/env python3
"""
PhysioFusion Model Training Simulator
Provides detailed error metrics, loss analysis, and tuning recommendations
"""

import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class ModelTrainingSimulator:
    """Advanced training simulator with detailed error analysis"""
    
    def __init__(self, model_name="physiofusion_ecg", verbose=True):
        self.model_name = model_name
        self.verbose = verbose
        self.history = None
        self.metrics = {}
        self.errors = {}
        self.warnings = []
        self.tuning_suggestions = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def create_training_data(self, num_samples=1000, input_dim=256):
        """Generate synthetic ECG training data"""
        if self.verbose:
            print(f"[DATA] Generating {num_samples} synthetic ECG samples (dim={input_dim})...")
        
        X = np.random.randn(num_samples, input_dim).astype(np.float32)
        y = np.random.rand(num_samples, 1).astype(np.float32)
        
        # Add some signal pattern
        X = X * 0.5 + np.sin(np.arange(input_dim) / 10)[np.newaxis, :]
        
        return X, y
    
    def create_model(self, input_dim=256, layers=None, dropout=0.2):
        """Create Keras model with configurable architecture"""
        if layers is None:
            layers = [128, 64, 32]
        
        if self.verbose:
            print(f"[MODEL] Creating model: input={input_dim}, hidden={layers}, dropout={dropout}")
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_dim,)))
        
        for units in layers:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
            model.add(tf.keras.layers.Dropout(dropout))
        
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        return model
    
    def analyze_errors(self, y_true, y_pred, dataset_name="validation"):
        """Comprehensive error analysis"""
        errors = {}
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        errors['MSE'] = float(mse)
        errors['MAE'] = float(mae)
        errors['RMSE'] = float(rmse)
        errors['R2'] = float(r2)
        
        # Residual analysis
        residuals = y_true - y_pred
        errors['residual_mean'] = float(np.mean(residuals))
        errors['residual_std'] = float(np.std(residuals))
        errors['residual_max'] = float(np.max(np.abs(residuals)))
        
        # Distribution analysis
        errors['pred_mean'] = float(np.mean(y_pred))
        errors['pred_std'] = float(np.std(y_pred))
        errors['pred_min'] = float(np.min(y_pred))
        errors['pred_max'] = float(np.max(y_pred))
        
        self.errors[dataset_name] = errors
        
        if self.verbose:
            print(f"\n[ERRORS - {dataset_name.upper()}]")
            print(f"  MSE:    {errors['MSE']:.6f}")
            print(f"  MAE:    {errors['MAE']:.6f}")
            print(f"  RMSE:   {errors['RMSE']:.6f}")
            print(f"  R2:     {errors['R2']:.6f}")
            print(f"  Residual Mean: {errors['residual_mean']:.6f}")
            print(f"  Residual Std:  {errors['residual_std']:.6f}")
            print(f"  Residual Max:  {errors['residual_max']:.6f}")
        
        return errors
    
    def check_data_quality(self, X, y, dataset_name="training"):
        """Check for data quality issues"""
        issues = []
        
        # Check for NaN
        if np.isnan(X).any():
            issues.append(f"NaN values found in X ({dataset_name})")
        if np.isnan(y).any():
            issues.append(f"NaN values found in y ({dataset_name})")
        
        # Check for Inf
        if np.isinf(X).any():
            issues.append(f"Inf values found in X ({dataset_name})")
        if np.isinf(y).any():
            issues.append(f"Inf values found in y ({dataset_name})")
        
        # Check class balance (for classification)
        if len(np.unique(y)) <= 10:
            unique, counts = np.unique(y, return_counts=True)
            min_count = np.min(counts)
            max_count = np.max(counts)
            if max_count / min_count > 5:
                issues.append(f"Class imbalance detected: {max_count/min_count:.1f}x")
        
        # Check feature scaling
        X_std = np.std(X, axis=0)
        if np.any(X_std < 0.1) or np.any(X_std > 10):
            issues.append("Feature scaling issue: features not normalized")
        
        return issues
    
    def generate_tuning_suggestions(self):
        """Generate tuning recommendations based on errors"""
        suggestions = []
        
        if 'validation' in self.errors:
            val_errors = self.errors['validation']
            
            # High error suggestions
            if val_errors['RMSE'] > 0.3:
                suggestions.append("[!] HIGH RMSE: Try increasing model capacity (more layers/units)")
            
            if val_errors['R2'] < 0.5:
                suggestions.append("[!] LOW R2: Model not capturing variance - consider deeper network")
            
            # Residual analysis
            if abs(val_errors['residual_mean']) > 0.1:
                suggestions.append("[!] BIASED RESIDUALS: Add bias regularization or more data")
            
            if val_errors['residual_std'] > 0.2:
                suggestions.append("[!] HIGH RESIDUAL VARIANCE: Increase regularization (L1/L2)")
            
            # Prediction range
            if val_errors['pred_min'] < -0.1 or val_errors['pred_max'] > 1.1:
                suggestions.append("[!] PREDICTION OUT OF RANGE: Adjust activation function or add clipping")
            
            # Good performance
            if val_errors['RMSE'] < 0.1 and val_errors['R2'] > 0.8:
                suggestions.append("[OK] MODEL PERFORMING WELL: Ready for quantization")
        
        self.tuning_suggestions = suggestions
        return suggestions
    
    def train_with_analysis(self, X_train, y_train, X_val, y_val, 
                            epochs=50, batch_size=32, learning_rate=0.001):
        """Train model with detailed analysis"""
        
        if self.verbose:
            print("\n[TRAINING] Starting model training with analysis...")
        
        # Check data quality
        train_issues = self.check_data_quality(X_train, y_train, "training")
        val_issues = self.check_data_quality(X_val, y_val, "validation")
        
        all_issues = train_issues + val_issues
        if all_issues:
            print("[DATA QUALITY ISSUES]")
            for issue in all_issues:
                print(f"  [!] {issue}")
                self.warnings.append(issue)
        
        # Create and compile model
        model = self.create_model(input_dim=X_train.shape[1])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        # Train with callbacks
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        print(f"\n[TRAINING] Running {epochs} epochs, batch_size={batch_size}...")
        self.history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1 if self.verbose else 0
        )
        
        # Evaluate
        if self.verbose:
            print("\n[EVALUATION] Analyzing predictions...")
        
        y_train_pred = model.predict(X_train, verbose=0)
        y_val_pred = model.predict(X_val, verbose=0)
        
        self.analyze_errors(y_train, y_train_pred, "training")
        self.analyze_errors(y_val, y_val_pred, "validation")
        
        # Generate suggestions
        self.generate_tuning_suggestions()
        
        if self.verbose:
            print("\n[SUGGESTIONS] Tuning recommendations:")
            for suggestion in self.tuning_suggestions:
                # Remove special characters for Windows compatibility
                clean_suggestion = suggestion.replace('⚠️', '[!]').replace('✓', '[OK]')
                print(f"  {clean_suggestion}")
        
        return model
    
    def generate_report(self, output_dir="./reports"):
        """Generate comprehensive training report"""
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            'timestamp': self.timestamp,
            'model_name': self.model_name,
            'errors': self.errors,
            'warnings': self.warnings,
            'tuning_suggestions': self.tuning_suggestions,
            'metrics': self.metrics
        }
        
        # Add training history
        if self.history:
            report['training_history'] = {
                'loss': [float(x) for x in self.history.history.get('loss', [])],
                'val_loss': [float(x) for x in self.history.history.get('val_loss', [])],
                'mae': [float(x) for x in self.history.history.get('mae', [])],
                'val_mae': [float(x) for x in self.history.history.get('val_mae', [])]
            }
        
        # Save JSON report
        report_path = os.path.join(output_dir, f"training_report_{self.timestamp}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        if self.verbose:
            print(f"\n[REPORT] Saved to {report_path}")
        
        # Create text summary
        summary_path = os.path.join(output_dir, f"training_summary_{self.timestamp}.txt")
        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("PhysioFusion Model Training Summary\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("TRAINING ERRORS\n")
            f.write("-" * 70 + "\n")
            if 'training' in self.errors:
                for key, val in self.errors['training'].items():
                    f.write(f"  {key:20s}: {val:.6f}\n")
            
            f.write("\nVALIDATION ERRORS\n")
            f.write("-" * 70 + "\n")
            if 'validation' in self.errors:
                for key, val in self.errors['validation'].items():
                    f.write(f"  {key:20s}: {val:.6f}\n")
            
            f.write("\nWARNINGS\n")
            f.write("-" * 70 + "\n")
            if self.warnings:
                for warning in self.warnings:
                    f.write(f"  ⚠️  {warning}\n")
            else:
                f.write("  None\n")
            
            f.write("\nTUNING SUGGESTIONS\n")
            f.write("-" * 70 + "\n")
            for suggestion in self.tuning_suggestions:
                f.write(f"  {suggestion}\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        if self.verbose:
            print(f"[REPORT] Saved to {summary_path}")
        
        return report_path, summary_path
    
    def plot_training_history(self, output_dir="./reports"):
        """Plot training history"""
        if not self.history:
            print("No training history available")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        # Loss plot
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE plot
        if 'mae' in self.history.history:
            axes[1].plot(self.history.history['mae'], label='Train MAE')
            axes[1].plot(self.history.history['val_mae'], label='Val MAE')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].set_title('Mean Absolute Error')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plot_path = os.path.join(output_dir, f"training_history_{self.timestamp}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        if self.verbose:
            print(f"[PLOT] Saved to {plot_path}")
        
        return plot_path


def main():
    """Demo of training simulator"""
    
    print("=" * 70)
    print("PhysioFusion Model Training Simulator")
    print("=" * 70)
    
    # Initialize simulator
    simulator = ModelTrainingSimulator(verbose=True)
    
    # Generate data
    X, y = simulator.create_training_data(num_samples=1000, input_dim=256)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Train with analysis
    print("\n" + "=" * 70)
    model = simulator.train_with_analysis(
        X_train, y_train, X_val, y_val,
        epochs=30,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Generate reports
    print("\n" + "=" * 70)
    report_json, report_txt = simulator.generate_report()
    
    # Plot
    print("\n" + "=" * 70)
    plot_path = simulator.plot_training_history()
    
    print("\n" + "=" * 70)
    print("Training simulation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
