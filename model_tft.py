import pandas as pd
import numpy as np
from fredapi import Fred
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer

def get_processed_data(api_key):
    fred = Fred(api_key=api_key)
    indicators = {'CPI': 'CPIAUCSL', 'FedFunds': 'FEDFUNDS', 'GDP': 'GDPC1', 
                  'Unemployment': 'UNRATE', 'Industrial_Production': 'INDPRO'}
    data_dict = {name: fred.get_series(code, observation_start='1960-01-01') for name, code in indicators.items()}
    df_cleaned = pd.DataFrame(data_dict).ffill()
    
    df_transformed = pd.DataFrame(index=df_cleaned.index)
    df_transformed['Inflation'] = np.log(df_cleaned['CPI']).diff()
    df_transformed['Rate_Change'] = df_cleaned['FedFunds'].diff()
    df_transformed['GDP_Growth'] = np.log(df_cleaned['GDP']).diff()
    df_transformed['Unemployment_Change'] = df_cleaned['Unemployment'].diff()
    df_transformed['Industrial_Production_Growth'] = np.log(df_cleaned['Industrial_Production']).diff()
    return df_transformed.dropna()

def create_forecasting_dataset(df):
    data = df.copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.reset_index(drop=True)

    data["series"] = "USA"
    data["time_idx"] = np.arange(len(data))
    
    max_prediction_length = 6
    max_encoder_length = 24
    training_cutoff = data["time_idx"].max() - max_prediction_length

    # Economic indicators as "known" variables 
    known_features = ["Rate_Change", "GDP_Growth", "Unemployment_Change", "Industrial_Production_Growth"]

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="Inflation",
        group_ids=["series"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=[],
        time_varying_known_reals=["time_idx"] + known_features,
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["Inflation"],
        target_normalizer=GroupNormalizer(groups=["series"], transformation=None),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, data, predict=False, stop_randomization=True)
    return training, validation


class CustomDataProcessor:
    def __init__(self, feature_cols, target_col):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.f_scaler = StandardScaler()
        self.t_scaler = StandardScaler()

    def prepare_loaders(self, df, split_ratio=0.8, batch_size=16):
        split_idx = int(len(df) * split_ratio)
        train_df, val_df = df.iloc[:split_idx], df.iloc[split_idx:]

        train_f = self.f_scaler.fit_transform(train_df[self.feature_cols])
        train_t = self.t_scaler.fit_transform(train_df[[self.target_col]])
        val_f = self.f_scaler.transform(val_df[self.feature_cols])
        val_t = self.t_scaler.transform(val_df[[self.target_col]])

        def create_sequences(features, target, window=24, horizon=6):
            X, y = [], []
            for i in range(len(features) - window - horizon + 1):
                X.append(features[i:i + window])
                y.append(target[i + window:i + window + horizon].flatten())
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(train_f, train_t)
        X_val, y_val = create_sequences(val_f, val_t)

        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                               torch.tensor(y_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                                             torch.tensor(y_val, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, (X_val, y_val)

    def inverse_transform_target(self, data):
        return self.t_scaler.inverse_transform(data.reshape(-1, 1)).reshape(data.shape)
