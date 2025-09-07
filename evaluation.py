"""
Model evaluation and visualization utilities for interest rate prediction.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive evaluation and visualization for interest rate models."""
    
    def __init__(self):
        self.evaluation_results = {}
        self.prediction_history = {}
        
    def evaluate_model_performance(self, predictor, df, target_col='overnight_rate', test_size=0.2):
        """
        Comprehensive model evaluation.
        
        Args:
            predictor: Trained InterestRatePredictor instance
            df (pd.DataFrame): Full dataset
            target_col (str): Target column name
            test_size (float): Proportion for testing
            
        Returns:
            dict: Evaluation results
        """
        print("Evaluating model performance...")
        
        # Get evaluation results from predictor
        results = predictor.evaluate_models(df, target_col, test_size)
        
        # additional metrics
        enhanced_results = {}
        for model_name, metrics in results.items():
            if 'error' not in metrics:
                enhanced_results[model_name] = {
                    **metrics,
                    'model_name': model_name,
                    'performance_grade': self._calculate_performance_grade(metrics['R2'])
                }
            else:
                enhanced_results[model_name] = metrics
        
        self.evaluation_results = enhanced_results
        return enhanced_results
    
    def _calculate_performance_grade(self, r2_score):
        """Calculate performance grade based on r2 score."""
        if r2_score >= 0.8:
            return 'A'
        elif r2_score >= 0.6:
            return 'B'
        elif r2_score >= 0.4:
            return 'C'
        elif r2_score >= 0.2:
            return 'D'
        else:
            return 'F'
    
    def create_evaluation_report(self, save_path=None):
        """Create a comprehensive evaluation report."""
        if not self.evaluation_results:
            print("No evaluation results available. Run evaluate_model_performance first.")
            return
        
        print("\n" + "="*60)
        print("MODEL EVALUATION REPORT")
        print("="*60)
        
        # resultd dataframe
        results_data = []
        for model_name, metrics in self.evaluation_results.items():
            if 'error' not in metrics:
                results_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'MAE': f"{metrics['MAE']:.4f}",
                    'RMSE': f"{metrics['RMSE']:.4f}",
                    'R²': f"{metrics['R2']:.4f}",
                    'Precision': f"{metrics.get('Precision', 0):.4f}",
                    'Accuracy': f"{metrics.get('Accuracy', 0):.4f}",
                    'Grade': metrics['performance_grade']
                })
        
        if results_data:
            results_df = pd.DataFrame(results_data)
            print(results_df.to_string(index=False))
            
            # Find best model
            best_model = max(self.evaluation_results.items(), 
                           key=lambda x: x[1].get('R2', -1) if 'error' not in x[1] else -1)
            print(f"\nBest performing model: {best_model[0].replace('_', ' ').title()}")
            print(f"R² Score: {best_model[1]['R2']:.4f}")
        
        print("="*60)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write("MODEL EVALUATION REPORT\n")
                f.write("="*60 + "\n")
                f.write(results_df.to_string(index=False))
                f.write(f"\n\nBest performing model: {best_model[0].replace('_', ' ').title()}\n")
                f.write(f"R² Score: {best_model[1]['R2']:.4f}\n")
            print(f"Report saved to: {save_path}")
    
    def plot_model_comparison(self, save_path=None):
        """Create text-based model comparison."""
        if not self.evaluation_results:
            print("No evaluation results available.")
            return
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)
        
        # prepare data
        models = []
        mae_scores = []
        rmse_scores = []
        r2_scores = []
        precision_scores = []
        accuracy_scores = []
        
        for model_name, metrics in self.evaluation_results.items():
            if 'error' not in metrics:
                models.append(model_name.replace('_', ' ').title())
                mae_scores.append(metrics['MAE'])
                rmse_scores.append(metrics['RMSE'])
                r2_scores.append(metrics['R2'])
                precision_scores.append(metrics.get('Precision', 0))
                accuracy_scores.append(metrics.get('Accuracy', 0))
        
        if not models:
            print("No valid model results to compare.")
            return
        
        print(f"{'Model':<15} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Precision':<10} {'Accuracy':<8}")
        print("-" * 60)
        
        for i, model in enumerate(models):
            print(f"{model:<15} {mae_scores[i]:<8.4f} {rmse_scores[i]:<8.4f} {r2_scores[i]:<8.4f} {precision_scores[i]:<10.4f} {accuracy_scores[i]:<8.4f}")
        
        # Find best model for each
        best_mae_idx = mae_scores.index(min(mae_scores))
        best_rmse_idx = rmse_scores.index(min(rmse_scores))
        best_r2_idx = r2_scores.index(max(r2_scores))
        best_precision_idx = precision_scores.index(max(precision_scores))
        best_accuracy_idx = accuracy_scores.index(max(accuracy_scores))
        
        print("\nBest Models:")
        print(f"  Lowest MAE:     {models[best_mae_idx]} ({mae_scores[best_mae_idx]:.4f})")
        print(f"  Lowest RMSE:    {models[best_rmse_idx]} ({rmse_scores[best_rmse_idx]:.4f})")
        print(f"  Highest R²:     {models[best_r2_idx]} ({r2_scores[best_r2_idx]:.4f})")
        print(f"  Highest Precision: {models[best_precision_idx]} ({precision_scores[best_precision_idx]:.4f})")
        print(f"  Highest Accuracy:  {models[best_accuracy_idx]} ({accuracy_scores[best_accuracy_idx]:.4f})")
        
        if save_path:
            with open(save_path.replace('.png', '.txt'), 'w') as f:
                f.write("MODEL PERFORMANCE COMPARISON\n")
                f.write("="*60 + "\n")
                f.write(f"{'Model':<15} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Precision':<10} {'Accuracy':<8}\n")
                f.write("-" * 60 + "\n")
                for i, model in enumerate(models):
                    f.write(f"{model:<15} {mae_scores[i]:<8.4f} {rmse_scores[i]:<8.4f} {r2_scores[i]:<8.4f} {precision_scores[i]:<10.4f} {accuracy_scores[i]:<8.4f}\n")
            print(f"Model comparison saved to: {save_path.replace('.png', '.txt')}")
        
        print("="*60)
    
    def plot_prediction_confidence(self, predictions_2025, save_path=None):
        """Display prediction confidence intervals as text."""
        if not predictions_2025:
            print("No predictions available.")
            return
        
        print("\n" + "="*60)
        print("2025 PREDICTIONS WITH CONFIDENCE INTERVALS")
        print("="*60)
        
        for date, result in predictions_2025.items():
            if result and result['ensemble'] is not None:
                # calculate confidence
                individual_values = list(result['individual_predictions'].values())
                if individual_values:
                    std_dev = np.std(individual_values)
                    confidence_interval = std_dev * 1.96  # 95% confidence
                else:
                    std_dev = 0
                    confidence_interval = 0
                
                print(f"{date.strftime('%B %d, %Y')}:")
                print(f"  Prediction: {result['ensemble']:.3f}%")
                print(f"  Standard Deviation: {std_dev:.3f}%")
                print(f"  95% Confidence Interval: ±{confidence_interval:.3f}%")
                print(f"  Range: {result['ensemble']-confidence_interval:.3f}% - {result['ensemble']+confidence_interval:.3f}%")
                print()
        
        if save_path:
            # Save as text file
            with open(save_path.replace('.png', '.txt'), 'w') as f:
                f.write("2025 PREDICTIONS WITH CONFIDENCE INTERVALS\n")
                f.write("="*60 + "\n")
                for date, result in predictions_2025.items():
                    if result and result['ensemble'] is not None:
                        individual_values = list(result['individual_predictions'].values())
                        if individual_values:
                            std_dev = np.std(individual_values)
                            confidence_interval = std_dev * 1.96
                        else:
                            std_dev = 0
                            confidence_interval = 0
                        
                        f.write(f"{date.strftime('%B %d, %Y')}:\n")
                        f.write(f"  Prediction: {result['ensemble']:.3f}%\n")
                        f.write(f"  Standard Deviation: {std_dev:.3f}%\n")
                        f.write(f"  95% Confidence Interval: ±{confidence_interval:.3f}%\n")
                        f.write(f"  Range: {result['ensemble']-confidence_interval:.3f}% - {result['ensemble']+confidence_interval:.3f}%\n\n")
            print(f"Prediction confidence saved to: {save_path.replace('.png', '.txt')}")
        
        print("="*60)
    
    def create_interactive_dashboard(self, df, predictions_2025, save_path=None):
        """Create a text-based dashboard summary."""
        print("Creating dashboard summary...")
        
        print("\n" + "="*80)
        print("BANK OF CANADA INTEREST RATE PREDICTION DASHBOARD")
        print("="*80)
        
        # Historical data summary
        print(f"\nHISTORICAL DATA SUMMARY:")
        print(f"  Data Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        print(f"  Total Data Points: {len(df)}")
        print(f"  Interest Rate Range: {df['overnight_rate'].min():.2f}% to {df['overnight_rate'].max():.2f}%")
        print(f"  Current Rate: {df['overnight_rate'].iloc[-1]:.2f}%")
        
        # Model performance summary
        if self.evaluation_results:
            print(f"\nMODEL PERFORMANCE SUMMARY:")
            for model_name, metrics in self.evaluation_results.items():
                if 'error' not in metrics:
                    print(f"  {model_name.replace('_', ' ').title()}:")
                    print(f"    R² Score: {metrics['R2']:.4f}")
                    print(f"    Accuracy: {metrics.get('Accuracy', 0):.4f}")
                    print(f"    Precision: {metrics.get('Precision', 0):.4f}")
        
        # 2025 predictions
        if predictions_2025:
            print(f"\n2025 PREDICTIONS SUMMARY:")
            for date, result in predictions_2025.items():
                if result and result['ensemble'] is not None:
                    print(f"  {date.strftime('%B %d, %Y')}: {result['ensemble']:.3f}%")
        
        # confidence
        if predictions_2025:
            print(f"\nCONFIDENCE ANALYSIS:")
            all_std_devs = []
            for date, result in predictions_2025.items():
                if result and result['individual_predictions']:
                    individual_values = list(result['individual_predictions'].values())
                    std_dev = np.std(individual_values)
                    all_std_devs.append(std_dev)
            
            if all_std_devs:
                avg_confidence = np.mean(all_std_devs)
                print(f"  Average Prediction Uncertainty: ±{avg_confidence:.3f}%")
                print(f"  Overall Confidence Level: {'High' if avg_confidence < 0.1 else 'Medium' if avg_confidence < 0.2 else 'Low'}")
        
        print("="*80)
        
        if save_path:
            # Save as text file
            with open(save_path.replace('.html', '.txt'), 'w') as f:
                f.write("BANK OF CANADA INTEREST RATE PREDICTION DASHBOARD\n")
                f.write("="*80 + "\n")
                f.write(f"\nHISTORICAL DATA SUMMARY:\n")
                f.write(f"  Data Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}\n")
                f.write(f"  Total Data Points: {len(df)}\n")
                f.write(f"  Interest Rate Range: {df['overnight_rate'].min():.2f}% to {df['overnight_rate'].max():.2f}%\n")
                f.write(f"  Current Rate: {df['overnight_rate'].iloc[-1]:.2f}%\n")
                
                if self.evaluation_results:
                    f.write(f"\nMODEL PERFORMANCE SUMMARY:\n")
                    for model_name, metrics in self.evaluation_results.items():
                        if 'error' not in metrics:
                            f.write(f"  {model_name.replace('_', ' ').title()}:\n")
                            f.write(f"    R² Score: {metrics['R2']:.4f}\n")
                            f.write(f"    Accuracy: {metrics.get('Accuracy', 0):.4f}\n")
                            f.write(f"    Precision: {metrics.get('Precision', 0):.4f}\n")
                
                if predictions_2025:
                    f.write(f"\n2025 PREDICTIONS SUMMARY:\n")
                    for date, result in predictions_2025.items():
                        if result and result['ensemble'] is not None:
                            f.write(f"  {date.strftime('%B %d, %Y')}: {result['ensemble']:.3f}%\n")
            print(f"Dashboard saved to: {save_path.replace('.html', '.txt')}")
    
    def generate_prediction_summary(self, predictions_2025):
        """Generate a summary of predictions for 2025."""
        if not predictions_2025:
            print("No predictions available.")
            return
        
        print("\n" + "="*60)
        print("2025 INTEREST RATE PREDICTIONS SUMMARY")
        print("="*60)
        
        for date, result in predictions_2025.items():
            if result and result['ensemble'] is not None:
                print(f"\n{date.strftime('%A, %B %d, %Y')}:")
                print(f"  Ensemble Prediction: {result['ensemble']:.3f}%")
                
                # individual model predictions
                print("  Individual Model Predictions:")
                for model_name, pred in result['individual_predictions'].items():
                    print(f"    {model_name.replace('_', ' ').title()}: {pred:.3f}%")
                
                # confidence
                individual_values = list(result['individual_predictions'].values())
                if individual_values:
                    std_dev = np.std(individual_values)
                    print(f"  Standard Deviation: {std_dev:.3f}%")
                    print(f"  Confidence Interval: ±{std_dev*1.96:.3f}% (95%)")
        
        print("="*60)

if __name__ == "__main__":

    from data_loader import BOCDataLoader
    from models import InterestRatePredictor
    
    # Load data and train models
    loader = BOCDataLoader()
    data = loader.load_boc_data()
    features = loader.prepare_features(data)
    
    predictor = InterestRatePredictor()
    predictor.train_all_models(features)
    
    # Make predictions
    predictions = predictor.predict_2025_dates(features)
    
    # Evaluate models
    evaluator = ModelEvaluator()
    evaluator.evaluate_model_performance(predictor, features)
    evaluator.create_evaluation_report()
    evaluator.generate_prediction_summary(predictions)
