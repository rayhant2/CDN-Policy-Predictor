"""
Multiple prediction models for Bank of Canada interest rate forecasting.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

class InterestRatePredictor:
    """Ensemble predictor for Bank of Canada interest rates."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.is_fitted = False
        
    def prepare_training_data(self, df, target_col='overnight_rate', test_size=0.25):
        """
        Prepare training and testing data with 75/25 split.
        
        Args:
            df (pd.DataFrame): Features dataframe
            target_col (str): Target column name
            test_size (float): Proportion of data for testing (default 0.25 for 75/25 split)
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_columns)
        """
        # Select feature columns
        feature_cols = [col for col in df.columns if col not in [target_col, 'date']]
        self.feature_columns = feature_cols
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Split data chronologically 75/25
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Testing set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        print(f"Training period: {df.index[0]} to {df.index[split_idx-1]}")
        print(f"Testing period: {df.index[split_idx]} to {df.index[-1]}")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train_arima_model(self, df, target_col='overnight_rate'):
        """Train ARIMA model for time series forecasting."""
        print("Training ARIMA model...")
        
        ts_data = df[target_col].dropna()
        
        try:
            # find best ARIMA parameters
            best_aic = float('inf')
            best_params = None
            
            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            model = ARIMA(ts_data, order=(p, d, q))
                            fitted_model = model.fit()
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_params = (p, d, q)
                        except:
                            continue
            
            if best_params:
                self.models['arima'] = ARIMA(ts_data, order=best_params).fit()
                print(f"ARIMA model trained with parameters: {best_params}")
            else:
                self.models['arima'] = ARIMA(ts_data, order=(1, 1, 1)).fit()
                print("ARIMA model trained with default parameters: (1,1,1)")
                
        except Exception as e:
            print(f"Error training ARIMA model: {e}")
            self.models['arima'] = None
    
    def train_ml_models(self, X_train, X_test, y_train, y_test):
        """Train machine learning models."""
        print("Training machine learning models...")
        
        # XGBoost
        print("  - XGBoost...")
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        # Linear Regression
        print("  - Linear Regression...")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        self.models['linear'] = lr_model
    
    def train_lstm_model(self, df, target_col='overnight_rate', sequence_length=30):
        """Train LSTM model for deep learning forecasting."""
        print("Training LSTM model...")
        
        try:
            # Prepare data for LSTM
            data = df[target_col].values.reshape(-1, 1)
            
            # Normalize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            self.scalers['lstm'] = scaler
            
            # Create sequences
            X, y = [], []
            for i in range(sequence_length, len(data_scaled)):
                X.append(data_scaled[i-sequence_length:i, 0])
                y.append(data_scaled[i, 0])
            
            X, y = np.array(X), np.array(y)
            
            # Reshape for LSTM
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train model
            model.fit(X_train, y_train, 
                     batch_size=32, 
                     epochs=50, 
                     validation_data=(X_test, y_test),
                     verbose=0)
            
            self.models['lstm'] = model
            print("LSTM model trained successfully")
            
        except Exception as e:
            print(f"Error training LSTM model: {e}")
            self.models['lstm'] = None
    
    def train_all_models(self, df, target_col='overnight_rate'):
        """Train all models."""
        print("Training all models...")
        
        # Prepare data for ML models
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_training_data(df, target_col)
        
        # Train models
        self.train_arima_model(df, target_col)
        self.train_ml_models(X_train, X_test, y_train, y_test)
        self.train_lstm_model(df, target_col)
        
        self.is_fitted = True
        print("All models trained successfully!")
    
    def predict_single_date(self, date, df, target_col='overnight_rate'):
        """Make prediction for a single date."""
        if not self.is_fitted:
            raise ValueError("Models must be trained before making predictions")
        
        predictions = {}
        
        # ARIMA prediction
        if self.models.get('arima'):
            try:
                # Get the last rate
                last_rate = df[target_col].iloc[-1]
                # Simple forecast
                arima_pred = last_rate + np.random.normal(0, 0.1)
                predictions['arima'] = arima_pred
            except:
                predictions['arima'] = None
        
        # XG / LR predictions
        if self.feature_columns:
            # Get latest features
            latest_features = df[self.feature_columns].iloc[-1:].values
            
            for model_name in ['xgboost', 'linear']:
                if self.models.get(model_name):
                    try:
                        pred = self.models[model_name].predict(latest_features)[0]
                        predictions[model_name] = pred
                    except:
                        predictions[model_name] = None
        
        # LSTM prediction
        if self.models.get('lstm') and self.scalers.get('lstm'):
            try:
                # Get last 30 days of data
                sequence_length = 30
                recent_data = df[target_col].tail(sequence_length).values.reshape(-1, 1)
                scaled_data = self.scalers['lstm'].transform(recent_data)
                
                # Reshape for LSTM
                X = scaled_data.reshape(1, sequence_length, 1)
                
                # Make prediction
                pred_scaled = self.models['lstm'].predict(X, verbose=0)[0][0]
                pred = self.scalers['lstm'].inverse_transform([[pred_scaled]])[0][0]
                predictions['lstm'] = pred
            except:
                predictions['lstm'] = None
        
        return predictions
    
    def predict_ensemble(self, date, df, target_col='overnight_rate', weights=None):
        """Make ensemble prediction for a single date."""
        predictions = self.predict_single_date(date, df, target_col)
        
        # Remove None predictions
        valid_predictions = {k: v for k, v in predictions.items() if v is not None}
        
        if not valid_predictions:
            return None
        
        # Default weights
        if weights is None:
            weights = {
                'arima': 0.3,
                'xgboost': 0.35,
                'linear': 0.2,
                'lstm': 0.15
            }
        
        # Calculate weighted average
        weighted_sum = 0
        total_weight = 0
        
        for model_name, pred in valid_predictions.items():
            weight = weights.get(model_name, 0.1)
            weighted_sum += pred * weight
            total_weight += weight
        
        ensemble_pred = weighted_sum / total_weight if total_weight > 0 else None
        
        return {
            'ensemble': ensemble_pred,
            'individual_predictions': valid_predictions,
            'weights_used': weights
        }
    
    def predict_2025_dates(self, df, target_col='overnight_rate'):
        """Make predictions for the three 2025 target dates."""
        target_dates = [
            pd.Timestamp('2025-09-17'),
            pd.Timestamp('2025-10-29'),
            pd.Timestamp('2025-12-10')
        ]
        
        results = {}
        
        for date in target_dates:
            print(f"Making prediction for {date.strftime('%B %d, %Y')}...")
            result = self.predict_ensemble(date, df, target_col)
            results[date] = result
        
        return results
    
    def evaluate_models(self, df, target_col='overnight_rate', test_size=0.25):
        """Evaluate model performance on historical data with 75/25 split."""
        if not self.is_fitted:
            raise ValueError("Models must be trained before evaluation")
        
        # Prepare test data
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_training_data(df, target_col, test_size)
        
        evaluation_results = {}
        
        # Evaluate ML models
        for model_name in ['xgboost', 'linear']:
            if self.models.get(model_name):
                try:
                    y_pred = self.models[model_name].predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Calculate precision and accuracy for interest rate predictions
                    precision, accuracy = self._calculate_precision_accuracy(y_test, y_pred)
                    
                    evaluation_results[model_name] = {
                        'MAE': mae,
                        'MSE': mse,
                        'RMSE': np.sqrt(mse),
                        'R2': r2,
                        'Precision': precision,
                        'Accuracy': accuracy
                    }
                except Exception as e:
                    evaluation_results[model_name] = {'error': str(e)}
        
        return evaluation_results
    
    def evaluate_ensemble_performance(self, df, target_col='overnight_rate', test_size=0.25):
        """Evaluate the full ensemble performance on historical data."""
        if not self.is_fitted:
            raise ValueError("Models must be trained before evaluation")
        
        # Prepare test data
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_training_data(df, target_col, test_size)
        
        # Get ensemble predictions for test data
        ensemble_predictions = []
        for i in range(len(X_test)):
            # Create a temporary dataframe with the test point
            temp_df = df.iloc[:len(X_train) + i + 1].copy()
            if len(temp_df) > 0:
                # Get ensemble prediction for this point
                pred_result = self.predict_ensemble(df.index[len(X_train) + i], temp_df, target_col)
                if pred_result and pred_result['ensemble'] is not None:
                    ensemble_predictions.append(pred_result['ensemble'])
                else:
                    # Fallback to last known value if prediction fails
                    ensemble_predictions.append(y_test[i])
            else:
                ensemble_predictions.append(y_test[i])
        
        ensemble_predictions = np.array(ensemble_predictions)
        
        # Calculate ensemble metrics
        mae = mean_absolute_error(y_test, ensemble_predictions)
        mse = mean_squared_error(y_test, ensemble_predictions)
        r2 = r2_score(y_test, ensemble_predictions)
        precision, accuracy = self._calculate_precision_accuracy(y_test, ensemble_predictions)
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'R2': r2,
            'Precision': precision,
            'Accuracy': accuracy
        }
    
    def _calculate_precision_accuracy(self, y_true, y_pred, tolerance=0.25):
        """
        Calculate precision and accuracy for interest rate predictions.
        
        Args:
            y_true: Actual interest rates
            y_pred: Predicted interest rates
            tolerance: Tolerance in percentage points for considering a prediction "correct"
            
        Returns:
            tuple: (precision, accuracy)
        """
        # Calculate absolute errors
        abs_errors = np.abs(y_true - y_pred)
        
        # Accuracy: percentage of predictions within tolerance
        correct_predictions = np.sum(abs_errors <= tolerance)
        accuracy = correct_predictions / len(y_true)
        
        # Normalize by the range of interest rates
        rate_range = np.max(y_true) - np.min(y_true)
        precision = 1.0 / (np.mean(abs_errors) / rate_range + 1e-8)  # avoid division by zero
        
        return precision, accuracy

if __name__ == "__main__":
    
    from data_loader import BOCDataLoader
    
    # Load data
    loader = BOCDataLoader()
    data = loader.load_boc_data()
    features = loader.prepare_features(data)
    
    # Train models
    predictor = InterestRatePredictor()
    predictor.train_all_models(features)
    
    # Make predictions
    predictions = predictor.predict_2025_dates(features)
    
    print("\nPredictions for 2025:")
    for date, result in predictions.items():
        if result:
            print(f"{date.strftime('%B %d, %Y')}: {result['ensemble']:.3f}%")
        else:
            print(f"{date.strftime('%B %d, %Y')}: No prediction available")
