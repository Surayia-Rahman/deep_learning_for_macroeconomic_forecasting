from statsmodels.tsa.api import VAR
import pandas as pd

def run_var_model(train_data, lags=12):
    """
    Fits a Vector Autoregression model.
    """
    model = VAR(train_data)
    results = model.fit(maxlags=lags, ic='aic')
    return results
