<br />
<div align="center">
  <h3 align="center">🪙 Canadian Monetary Policy Rate Predictor</h3>

  <p align="center">
    Forecasting the Bank of Canada's policy rate changes to drive mortgage interest rate strategy
  </p>
</div>
<br />


# About The Project

This is a comprehensive machine learning system that predicts Bank of Canada (BOC) interest rate announceements using an ensemble model, for the following 2025 BOC announcement dates:

- **Wednesday, September 17, 2025**
- **Wednesday, October 29, 2025** 
- **Wednesday, December 10, 2025**


## Architecture

The predictor uses an **ensemble learning approach** by combining the following architectures:

- **ARIMA**: Statistical, time-series forecasting
- **LSTM**: time-series forecasting, pattern identification
- **XGBoost**: Gradient boosting
- **Linear Regression**: Baseline linear model

### Feature Engineering
- Rolling statistics (7, 14, 30, 90-day windows)
- Technical indicators (rate changes, volatility)
- Time-based features (announcement cycles)
- Economic indicators (inflation, GDP, unemployment)

## 📊 Model Performance

### Performance Metrics
- **80.73% Accuracy**
- **14.28 Precision**
- **95.28% R² Score**
- **0.16% MAE**

### 🔮 2025 Policy Rate Predictions
- **September 17, 2025**: *3.187%*  
- **October 29, 2025**: *3.178%*  
- **December 10, 2025**: *3.222%* 

### Data Split
- **75/25 train-test split** for balanced evaluation  
- **522 total data points** (2015–2025)  
- **109 test samples** for robust validation 
<br />

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/rayhant2/CDN-Policy-Predictor.git
cd CDN-Policy-Predictor

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run with the data from the dataset
python main.py

```

## 📋 Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- xgboost, tensorflow

## 📄 License

MIT License - see LICENSE file for details.