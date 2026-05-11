"""
Comparative Analysis of Recurrent and Attention-Based Deep Learning Architectures for Multivariate Multi-Horizon Macroeconomic Forecasting**

`[Nimrah Adam, Jakia Nowshin, Surayia Rahman]
"""

# Install necessary libraries
# fredapi
# pytorch-forecasting
# pytorch-lightning


import os
import sys
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import random
from IPython.display import display

# Add pytorch_forecasting imports needed for the models
import pytorch_forecasting
from pytorch_forecasting.models.rnn import RecurrentNetwork
from pytorch_forecasting import TemporalFusionTransformer

# Path Setup
project_path = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_path, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Module Imports
from src import data_utils
from src import evaluation
from src import visuals
from src import trainer
from src import model_var
from src import model_library_rnn
from src import model_tft
from src import model_custom_lstm
from src import model_custom_transformer

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variables for full reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # Enable deterministic algorithms in PyTorch
    torch.use_deterministic_algorithms(True, warn_only=True)

def main():
    # Set the seed at the very beginning
    seed_everything(42)

    # data recovery
    API_KEY = os.getenv("FRED_API_KEY")
    if API_KEY is None:
        print("Warning: FRED_API_KEY not found. Please set it as an environment variable. Data loading may fail.")
        print("Skipping data fetch due to missing API key.")
        return 
        
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process data for Library models
    df_transformed = data_utils.get_processed_data(API_KEY)
    training_dataset, validation = data_utils.create_forecasting_dataset(df_transformed)
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=32, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=320, num_workers=0)

    # Process data for Custom models
    processor = data_utils.CustomDataProcessor(
        feature_cols=["Inflation", "Rate_Change", "GDP_Growth", "Unemployment_Change", "Industrial_Production_Growth"],
        target_col="Inflation"
    )
    c_train_loader, c_val_loader, (X_val, y_val) = processor.prepare_loaders(df_transformed)

    # model training and prediction

    # VAR
    var_data = df_transformed[['Inflation', 'Rate_Change', 'GDP_Growth', 'Unemployment_Change', 'Industrial_Production_Growth']]
    train_var, test_var = var_data.iloc[:-6], var_data.iloc[-6:]
    res_var_model = model_var.run_var_model(train_var)
    var_forecast = res_var_model.forecast(train_var.values[-res_var_model.k_ar:], steps=6)

    # Library GRU
    features = ["Inflation", "Rate_Change", "GDP_Growth", "Unemployment_Change", "Industrial_Production_Growth"]
    gru_model = RecurrentNetwork.from_dataset(
        training_dataset, cell_type="GRU", hidden_size=32, rnn_layers=2, dropout=0.2,
        loss=pytorch_forecasting.metrics.RMSE(), learning_rate=1e-3
    )
    pl_trainer = pl.Trainer(max_epochs=50, accelerator="cpu", enable_checkpointing=False, logger=False, deterministic=True)
    pl_trainer.fit(model_library_rnn.RNN_Wrapper(gru_model), train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    gru_preds = gru_model.predict(val_dataloader, mode="prediction", return_x=False)
    if isinstance(gru_preds, torch.Tensor):
        gru_preds = gru_preds.detach().cpu().numpy()

    # Library LSTM
    lstm_lib_model = RecurrentNetwork.from_dataset(
        training_dataset, cell_type="LSTM", hidden_size=32, rnn_layers=2, dropout=0.2,
        loss=pytorch_forecasting.metrics.RMSE(), learning_rate=1e-3
    )
    pl_trainer = pl.Trainer(max_epochs=50, accelerator="cpu", enable_checkpointing=False, logger=False, deterministic=True)
    pl_trainer.fit(model_library_rnn.RNN_Wrapper(lstm_lib_model), train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    lib_lstm_preds = lstm_lib_model.predict(val_dataloader, mode="prediction", return_x=False)
    if isinstance(lib_lstm_preds, torch.Tensor):
        lib_lstm_preds = lib_lstm_preds.detach().cpu().numpy()

    # TFT
    tft_model = TemporalFusionTransformer.from_dataset(
        training_dataset, learning_rate=0.03, hidden_size=16, attention_head_size=4,
        dropout=0.1, hidden_continuous_size=8, loss=pytorch_forecasting.metrics.RMSE()
    )
    pl_trainer = pl.Trainer(max_epochs=50, accelerator="cpu", enable_checkpointing=False, logger=False, deterministic=True)
    pl_trainer.fit(model_tft.TFT_Wrapper(tft_model), train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    tft_preds = tft_model.predict(val_dataloader, mode="prediction", return_x=False)
    if isinstance(tft_preds, torch.Tensor):
        tft_preds = tft_preds.detach().cpu().numpy()

    # Custom LSTM
    raw_lstm = model_custom_lstm.CustomLSTM(input_size=5, hidden_size=16, output_size=6).to(device)
    trained_custom_lstm = trainer.train_custom_model(raw_lstm, c_train_loader, c_val_loader, device)
    trained_custom_lstm.eval()
    with torch.no_grad():
        c_lstm_preds_raw = trained_custom_lstm(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy()
    c_lstm_preds = processor.inverse_transform_target(c_lstm_preds_raw)
    actuals_inv = processor.inverse_transform_target(y_val)

    # Custom Transformer
    raw_trans = model_custom_transformer.CustomTransformer(input_size=5, d_model=32, nhead=4, num_layers=2, output_size=6).to(device)
    trained_custom_trans = trainer.train_custom_model(raw_trans, c_train_loader, c_val_loader, device)
    trained_custom_trans.eval()
    with torch.no_grad():
        c_trans_preds_raw = trained_custom_trans(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy()
    c_trans_preds = processor.inverse_transform_target(c_trans_preds_raw)

    # metrics
    lib_actuals_list = []
    for batch in val_dataloader:
        x, y = batch
        lib_actuals_list.append(y[0])

    lib_actuals = torch.cat(lib_actuals_list).cpu().numpy()
    
    # Filter out NaN values
    valid_mask = ~(np.isnan(gru_preds).any(axis=1) | np.isnan(lib_lstm_preds).any(axis=1) | 
                   np.isnan(tft_preds).any(axis=1) | np.isnan(lib_actuals).any(axis=1))
    
    gru_preds = gru_preds[valid_mask]
    lib_lstm_preds = lib_lstm_preds[valid_mask]
    tft_preds = tft_preds[valid_mask]
    lib_actuals = lib_actuals[valid_mask]

    results = [
        evaluation.calculate_metrics("VAR", test_var['Inflation'].values, var_forecast[:, 0]),
        evaluation.calculate_metrics("GRU", lib_actuals, gru_preds),
        evaluation.calculate_metrics("Library LSTM", lib_actuals, lib_lstm_preds),
        evaluation.calculate_metrics("TFT", lib_actuals, tft_preds),
        evaluation.calculate_metrics("Custom LSTM", actuals_inv, c_lstm_preds),
        evaluation.calculate_metrics("Custom Transformer", actuals_inv, c_trans_preds)
    ]

    # output

    # 1. Final Ranked Table
    df_results = pd.DataFrame(results).sort_values(by="Direction", ascending=False).reset_index(drop=True)
    df_results.index += 1
    df_results.index.name = 'Rank'

    print("\nModel performance ranking (By Directional Accuracy)")
    print("-" * 65)
    print(df_results.to_string(formatters={'RMSE': '{:.4f}'.format, 'MAE': '{:.4f}'.format, 'Direction': '{:.4f}'.format}))

    # Metrics Visualization
    visuals.plot_metrics_comparison(results)

if __name__ == "__main__":
    main()

"""
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
"""