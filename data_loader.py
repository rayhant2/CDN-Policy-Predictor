"""
Data loading and preprocessing utilities for Bank of Canada interest rate prediction.
"""

import pandas as pd
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from io import StringIO

class BOCDataLoader:
    """Load and preprocess Bank of Canada interest rate and economic data."""
    
    def __init__(self):
        """Initialize the data loader."""
        pass
            
    def load_boc_data(self, file_path=None):
        """
        Load BOC interest rate data from CSV file.
        
        Args:
            file_path (str): Path to BOC data CSV file
            
        Returns:
            pd.DataFrame: Processed BOC data
        """
        if file_path:
            # Load from user-provided file
            print(f"Loading BOC data from: {file_path}")
            df = self._load_boc_csv(file_path)
        else:
            # Use default CSV file
            default_csv = "lookup-2.csv"
            if os.path.exists(default_csv):
                print(f"Using default data file: {default_csv}")
                df = self._load_boc_csv(default_csv)
            else:
                raise FileNotFoundError("No data file provided and lookup-2.csv not found. Please provide a data file.")
            
        return df.sort_index()
    
    def _load_boc_csv(self, file_path):
        """
        Load and process the specific BOC CSV format.
        
        Args:
            file_path (str): Path to BOC CSV file
            
        Returns:
            pd.DataFrame: Processed BOC data
        """
        # Find the first dataset
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find the start / Bank rate data
        data_start = None
        for i, line in enumerate(lines):
            if line.strip() == 'Date,V80691310':
                data_start = i + 1 
                break
        
        if data_start is None:
            raise ValueError("Could not find the Bank rate data section in the CSV file")
        
        # Find end of the first dataset
        data_end = None
        for i in range(data_start, len(lines)):
            line = lines[i].strip()
            if line.startswith('Weekly (') and 'series:' in line and i > data_start + 10:
                data_end = i
                break
        
        if data_end is None:
            data_end = len(lines)
        
        # Extract just the Bank rate data
        data_lines = lines[data_start:data_end]
        
        # Create a temporary CSV
        temp_csv_content = 'Date,V80691310\n' + ''.join(data_lines)
        
        # Read the data
        df = pd.read_csv(StringIO(temp_csv_content))
        df.columns = df.columns.str.strip()
        
        if 'V80691310' in df.columns:
            df = df.rename(columns={'V80691310': 'overnight_rate'})
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        
        # Remove rows with missing data
        df = df.dropna()
        
        df['overnight_rate'] = pd.to_numeric(df['overnight_rate'], errors='coerce')
        df = df.dropna()
        
        print(f"Loaded {len(df)} data points from {df.index.min()} to {df.index.max()}")
        print(f"Interest rate range: {df['overnight_rate'].min():.2f}% to {df['overnight_rate'].max():.2f}%")
        
        return df
    
    def prepare_features(self, df, lookback_days=30):
        """
        Prepare features for machine learning models.
        
        Args:
            df (pd.DataFrame): Input data
            lookback_days (int): Number of days to look back for features
            
        Returns:
            pd.DataFrame: Features for ML models
        """
        features_df = df.copy()
        
        # Create lagged features
        for lag in range(1, lookback_days + 1):
            features_df[f'rate_lag_{lag}'] = features_df['overnight_rate'].shift(lag)
            
        # Create rolling statistics
        for window in [7, 14, 30, 90]:
            features_df[f'rate_ma_{window}'] = features_df['overnight_rate'].rolling(window).mean()
            features_df[f'rate_std_{window}'] = features_df['overnight_rate'].rolling(window).std()
            
        # Create technical indicators
        features_df['rate_change'] = features_df['overnight_rate'].diff()
        features_df['rate_change_pct'] = features_df['overnight_rate'].pct_change()
        
        # Create time-based features
        features_df['month'] = features_df.index.month
        features_df['quarter'] = features_df.index.quarter
        features_df['year'] = features_df.index.year
        features_df['day_of_year'] = features_df.index.dayofyear
        features_df['is_month_end'] = features_df.index.is_month_end.astype(int)
        
        # Create announcement date features (approximate BOC schedule)
        features_df['is_announcement_month'] = features_df['month'].isin([1, 3, 4, 6, 7, 9, 10, 12]).astype(int)
        features_df['days_since_last_announcement'] = self._calculate_days_since_announcement(features_df.index)
        
        return features_df.dropna()
    
    def _calculate_days_since_announcement(self, dates):
        """Calculate days since last potential announcement."""
        announcement_months = [1, 3, 4, 6, 7, 9, 10, 12]
        days_since = []
        
        for date in dates:
            # Find last announcement month
            last_announcement = None
            for i in range(12):
                check_date = date - timedelta(days=30*i)
                if check_date.month in announcement_months:
                    last_announcement = check_date
                    break
            
            if last_announcement:
                days_since.append((date - last_announcement).days)
            else:
                days_since.append(0)
                
        return days_since
    
    def get_target_dates_2025(self):
        """Get the specific target dates for 2025 predictions."""
        return [
            datetime(2025, 9, 17),   # Wednesday, September 17
            datetime(2025, 10, 29),  # Wednesday, October 29  
            datetime(2025, 12, 10)   # Wednesday, December 10
        ]

if __name__ == "__main__":
    loader = BOCDataLoader()
    data = loader.load_boc_data()
    features = loader.prepare_features(data)
    
    print("Data shape:", data.shape)
    print("Features shape:", features.shape)
    print("\nSample data:")
    print(data.head())
    print("\nTarget dates for 2025:")
    for date in loader.get_target_dates_2025():
        print(f"  {date.strftime('%A, %B %d, %Y')}")
