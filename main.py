"""
Main script for Bank of Canada interest rate prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import os

from data_loader import BOCDataLoader
from models import InterestRatePredictor
from evaluation import ModelEvaluator

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Bank of Canada Interest Rate Predictor')
    parser.add_argument('--data-file', type=str, help='Path to BOC data CSV file')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory for results')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training (use existing models)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Bank of Canada Interest Rate Predictor")
    print("=" * 50)
    
    # Initialize data loader
    print("\nLoading data...")
    loader = BOCDataLoader()
    
    if args.data_file:
        print(f"Loading data from: {args.data_file}")
        data = loader.load_boc_data(args.data_file)
    else:
        default_csv = "lookup-2.csv"
        if os.path.exists(default_csv):
            print(f"Using default data file: {default_csv}")
            data = loader.load_boc_data(default_csv)
        else:
            print("No data file provided.")
            data = loader.load_boc_data()
    
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    # Prepare features
    print("\n🔧 Preparing features...")
    features = loader.prepare_features(data)
    print(f"Features shape: {features.shape}")
    
    # Initialize predictor
    predictor = InterestRatePredictor()
    
    if not args.skip_training:
        # Train models
        print("\n🤖 Training models...")
        predictor.train_all_models(features)
        
        print("Models trained successfully!")
    else:
        print("Skipping model training...")
    
    # Make predictions for 2025
    print("\nMaking predictions for 2025...")
    predictions_2025 = predictor.predict_2025_dates(features)
    
    evaluator = ModelEvaluator()
    
    if not args.skip_training:
        # Evaluate models
        print("\nEvaluating model performance...")
        evaluator.evaluate_model_performance(predictor, features)
        
        print("\nGenerating evaluation report...")
        evaluator.create_evaluation_report(os.path.join(args.output_dir, 'evaluation_report.txt'))
        
        print("\nCreating visualizations...")
        evaluator.plot_model_comparison(os.path.join(args.output_dir, 'model_comparison.png'))
    
    # Generate prediction summary
    print("\nGenerating prediction summary...")
    evaluator.generate_prediction_summary(predictions_2025)
    
    # Create interactive dashboard
    print("\nCreating interactive dashboard...")
    evaluator.create_interactive_dashboard(
        data, 
        predictions_2025, 
        os.path.join(args.output_dir, 'prediction_dashboard.html')
    )
    
    # Save predictions to CSV
    print("\nSaving predictions...")
    predictions_df = []
    for date, result in predictions_2025.items():
        if result and result['ensemble'] is not None:
            row = {
                'date': date,
                'ensemble_prediction': result['ensemble'],
                'confidence_std': np.std(list(result['individual_predictions'].values())) if result['individual_predictions'] else 0
            }
            for model_name, pred in result['individual_predictions'].items():
                row[f'{model_name}_prediction'] = pred
            predictions_df.append(row)
    
    if predictions_df:
        predictions_df = pd.DataFrame(predictions_df)
        predictions_df.to_csv(os.path.join(args.output_dir, '2025_predictions.csv'), index=False)
        print(f"Predictions saved to: {os.path.join(args.output_dir, '2025_predictions.csv')}")
    
    # Final results
    print("\n" + "=" * 60)
    print("🎯 FINAL PREDICTIONS FOR 2025")
    print("=" * 60)
    
    for date, result in predictions_2025.items():
        if result and result['ensemble'] is not None:
            print(f"{date.strftime('%A, %B %d, %Y')}: {result['ensemble']:.3f}%")
    
    print("\n✅ Analysis complete! Check the output directory for detailed results.")
    print(f"📁 Output directory: {args.output_dir}")

def run_quick_demo():
    """Run a quick demonstration with your BOC data."""
    print("🚀 Running quick demonstration with your BOC data...")
    
    loader = BOCDataLoader()
    csv_file = "lookup-2.csv"
    if os.path.exists(csv_file):
        data = loader.load_boc_data(csv_file)
    else:
        print("CSV file not found, using sample data...")
        data = loader.load_boc_data()
    
    features = loader.prepare_features(data)
    
    # Train models
    predictor = InterestRatePredictor()
    predictor.train_all_models(features)
    
    # Evaluate ensemble performance and display precision/accuracy
    print("\n📊 Ensemble Performance Evaluation:")
    print("-" * 50)
    ensemble_metrics = predictor.evaluate_ensemble_performance(features)
    
    # Display ensemble precision and accuracy
    print(f"Ensemble Precision: {ensemble_metrics['Precision']:.4f}")
    print(f"Ensemble Accuracy:  {ensemble_metrics['Accuracy']:.4f}")
    print(f"Ensemble R² Score:  {ensemble_metrics['R2']:.4f}")
    print(f"Mean Absolute Error: {ensemble_metrics['MAE']:.4f}")
    print()
    
    # Make predictions
    predictions = predictor.predict_2025_dates(features)
    
    # Display results
    print("\n🎯 2025 Interest Rate Predictions:")
    print("-" * 40)
    for date, result in predictions.items():
        if result and result['ensemble'] is not None:
            print(f"{date.strftime('%B %d, %Y')}: {result['ensemble']:.3f}%")
    
    return predictions

if __name__ == "__main__":
    
    run_quick_demo()
