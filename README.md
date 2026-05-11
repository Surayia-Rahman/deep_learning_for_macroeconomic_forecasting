# Deep Learning for Macroeconomic Forecasting

This project implements and compares **custom-built deep learning architectures** (vanilla Transformer and LSTM) against **library-based models** (GRU, LSTM, TFT) for multivariate, multi-horizon macroeconomic forecasting, with a focus on predicting inflation dynamics using U.S. economic indicators.

---

## Overview

The pipeline builds a complete forecasting system:

* Retrieves macroeconomic time-series data from the FRED API
* Applies economic transformations (first differences for rates, growth rates for levels)
* Uses PyTorch Forecasting's TimeSeriesDataSet for sequence preparation
* Trains and compares six models:
  * **Library GRU** (pytorch-forecasting RecurrentNetwork)
  * **Library LSTM** (pytorch-forecasting RecurrentNetwork)
  * **Temporal Fusion Transformer (TFT)** (pytorch-forecasting)
  * **Vector Autoregression (VAR)** (statistical baseline)
  * **Custom LSTM** (from scratch)
  * **Custom Transformer** (vanilla encoder architecture)
* Evaluates performance using RMSE, MAE, and directional accuracy

The primary objective is to analyze whether **custom implementations of sequence models** can match or outperform **library-optimized architectures**, and to understand trade-offs in learning dynamics, generalization, and stability in macroeconomic time-series settings.

---

## Project Structure

```bash
DS8013_project_codes_Adam_Nowshin_Rahman/
│
├── dl_project_macroeconomic_forecasting_final.py  # Main script
├── README.md
│
└── src/
    ├── __init__.py
    ├── data_utils.py              # FRED data fetching & TimeSeriesDataSet creation
    ├── trainer.py                 # Custom model training logic
    ├── evaluation.py              # Metrics: RMSE, MAE, directional accuracy
    ├── visuals.py                 # Plotting functions
    ├── model_var.py               # Vector Autoregression model
    ├── model_library_rnn.py       # PyTorch Lightning wrapper for library models
    ├── model_custom_lstm.py       # Custom LSTM implementation
    ├── model_custom_transformer.py # Custom Transformer implementation
    └── model_tft.py               # TFT model utilities
```

---

## Features

### Data Pipeline

* Data is retrieved using `fredapi` from the Federal Reserve Economic Data (FRED)
* **Macroeconomic indicators** (monthly):
  * **CPIAUCSL** - Consumer Price Index (Inflation)
  * **DFF** - Federal Funds Effective Rate (Interest Rate)
  * **GDP** - Gross Domestic Product
  * **UNRATE** - Unemployment Rate
  * **INDPRO** - Industrial Production Index

### Feature Engineering

* **Target variable**: Inflation (first difference of CPI)
* **Known reals** (available in decoder during prediction):
  * Rate_Change: First difference of Federal Funds Rate
  * GDP_Growth: Growth rate of GDP
  * Unemployment_Change: First difference of Unemployment Rate
  * Industrial_Production_Growth: Growth rate of Industrial Production

* All features are made stationary through differencing/growth rate calculations

### Sequence Configuration

* **Encoder length**: 24 timesteps (lookback window)
* **Decoder length**: 6 timesteps (forecast horizon)
* **Min encoder length**: 12 timesteps
* **Min prediction length**: 1 timestep
* Uses TimeSeriesDataSet from pytorch-forecasting with:
  * `time_varying_known_reals`: Economic indicators (decoder-available)
  * `time_varying_unknown_reals`: Inflation (target only)
  * Additional features: relative time index, target scales, encoder length

### Models Implemented

1. **Library GRU** (RecurrentNetwork with GRU cell)
   - Hidden size: 32, Layers: 2, Dropout: 0.2
   
2. **Library LSTM** (RecurrentNetwork with LSTM cell)
   - Hidden size: 32, Layers: 2, Dropout: 0.2
   
3. **Temporal Fusion Transformer (TFT)**
   - Hidden size: 32, Attention heads: 4, Dropout: 0.1
   
4. **Vector Autoregression (VAR)**
   - Classical statistical baseline using statsmodels
   
5. **Custom LSTM** (manual PyTorch implementation)
   - Custom architecture with LSTM cells
   
6. **Custom Transformer** (vanilla encoder architecture)
   - Positional encoding, multi-head attention

### Training Framework

* **Framework**: PyTorch Lightning for library models, custom training loop for custom models
* **Loss Function**: RMSE (Root Mean Squared Error)
* **Optimizer**: Adam (learning rate: 1e-3)
* **Epochs**: 50
* **Gradient clipping**: Enabled in custom models
* **Random seed**: 42 (for reproducibility)

---

## Installation

### Required Packages

```bash
pip install pandas numpy torch scikit-learn fredapi matplotlib seaborn
pip install pytorch-forecasting lightning statsmodels
```

### Package Versions (Tested)

```
torch==2.10.0
pytorch-forecasting==1.7.0
lightning==2.6.1
numpy==2.0.2
pandas==2.2.2
statsmodels>=0.14.0
fredapi>=0.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

---

## FRED API Setup

You need a FRED API key to fetch economic data:

1. Register at: https://fred.stlouisfed.org/
2. Get your API key from your account
3. Set it as an environment variable:

```bash
export FRED_API_KEY="your_api_key_here"
```

Or pass it directly in the script (see Usage section).

---

## Usage

### Running the Main Script

```bash
# Set API key and run
export FRED_API_KEY="your_api_key_here"
python3 dl_project_macroeconomic_forecasting_final.py
```

### What the Script Does

1. **Fetches data** from FRED API for the five economic indicators
2. **Transforms data** into stationary features (growth rates, differences)
3. **Creates datasets** using TimeSeriesDataSet for training and validation
4. **Trains six models**:
   - Library GRU
   - Library LSTM
   - TFT (Temporal Fusion Transformer)
   - VAR (Vector Autoregression)
   - Custom LSTM
   - Custom Transformer
5. **Evaluates models** using:
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - Directional Accuracy (percentage of correct trend predictions)
6. **Generates visualizations**:
   - Performance comparison bar charts
   - Saved to `model_performance_comparison.png`
7. **Prints ranking** of models by directional accuracy

### Expected Output

```
Model performance ranking (By Directional Accuracy)
-----------------------------------------------------------------
                   Model   RMSE    MAE Direction
Rank                                            
1                    TFT 0.0016 0.0012    0.6292
2                    GRU 0.0017 0.0012    0.6109
3                    VAR 0.0028 0.0020    0.6000
4           Library LSTM 0.0017 0.0012    0.5975
5            Custom LSTM 0.0026 0.0019    0.5015
6     Custom Transformer 0.0026 0.0019    0.4846
```

---

## Evaluation Metrics

### RMSE (Root Mean Squared Error)
Measures the average magnitude of prediction errors, with larger errors penalized more heavily.

### MAE (Mean Absolute Error)
Measures the average absolute difference between predictions and actual values.

### Directional Accuracy
Percentage of time the model correctly predicts the direction of change (up or down) across the 6-step forecast horizon. This is particularly important for economic forecasting where trend direction matters.


## Key Insights

* **Library models** (GRU, LSTM, TFT) are strong performers with optimized implementations
* **TFT** shows excellent performance with attention mechanisms for multivariate forecasting
* **VAR** provides a solid statistical baseline, matching or exceeding some neural models
* **Custom LSTM** can be competitive but requires careful tuning
* **Custom Transformers** are sensitive to data size and hyperparameters in low-data regimes
* **Directional accuracy** is crucial for economic forecasting applications

---

## Limitations

* **Limited data size**: Monthly macroeconomic data has fewer samples than typical deep learning tasks
* **Computational constraints**: CPU-only training for some models
* **Hyperparameter tuning**: Limited grid search due to training time
* **Feature engineering**: Basic transformations; could benefit from more domain expertise
* **Stationarity assumptions**: Assumes transformations achieve stationarity
* **Single target**: Only forecasts inflation; could extend to multi-target forecasting

---

## Future Work

* **Hyperparameter optimization**: Systematic grid search or Bayesian optimization
* **More macroeconomic indicators**: Include more FRED series (housing, trade, financial markets)
* **Probabilistic forecasting**: Implement quantile regression or distributional outputs
* **Hybrid models**: Combine econometric models (VAR, ARIMA) with neural networks
* **Attention visualization**: Analyze TFT attention weights to interpret feature importance
* **Multi-target forecasting**: Predict multiple economic indicators simultaneously
* **Longer horizons**: Extend prediction window beyond 6 months
* **Real-time updates**: Build pipeline for continuous data fetching and retraining
* **Ensemble methods**: Combine predictions from multiple models

---

## Project Files

* **dl_project_macroeconomic_forecasting_final.py**: Main execution script
* **src/data_utils.py**: FRED data fetching, transformations, TimeSeriesDataSet creation
* **src/trainer.py**: Custom model training loops
* **src/evaluation.py**: Metrics calculation (RMSE, MAE, directional accuracy)
* **src/visuals.py**: Plotting functions for performance comparison
* **src/model_var.py**: Vector Autoregression implementation
* **src/model_library_rnn.py**: PyTorch Lightning wrapper for library models
* **src/model_custom_lstm.py**: Custom LSTM architecture
* **src/model_custom_transformer.py**: Custom Transformer architecture
* **src/model_tft.py**: TFT model utilities

---

## Troubleshooting

### Import Errors
Make sure all packages are installed with correct versions:
```bash
pip install torch==2.10.0 pytorch-forecasting==1.7.0 lightning==2.6.1
```

### FRED API Errors
- Check your API key is valid
- Ensure environment variable is set: `echo $FRED_API_KEY`
- Verify internet connection for API access

### Memory Issues
- Reduce batch size in data_utils.py
- Use CPU-only mode if GPU memory is limited

### NaN Values in Predictions
- The script automatically filters NaN samples before evaluation
- Check data quality if excessive NaN values occur

---

## References

* **FRED API**: Federal Reserve Economic Data (https://fred.stlouisfed.org/)
* **PyTorch Forecasting**: Time series forecasting with PyTorch (https://pytorch-forecasting.readthedocs.io/)
* **PyTorch Lightning**: Lightweight PyTorch wrapper (https://lightning.ai/)
* **Temporal Fusion Transformer**: Lim et al., 2021 (https://arxiv.org/abs/1912.09363)

---

## License

This project is for educational purposes as part of DS8013 Deep Learning course.

---

## Authors

Adam, Nowshin, Rahman  
Winter 2026  
DS8013 - Deep Learning

Academic and research use only.
