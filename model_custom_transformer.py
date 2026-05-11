import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def directional_accuracy_multistep(preds, actuals):
    """
    Computes directional accuracy across all forecast windows.
    """
    # Ensure inputs are numpy arrays
    preds = np.array(preds)
    actuals = np.array(actuals)
    
    correct = 0
    total = 0

    # If 1D (flattened), handle as a single sequence
    if len(preds.shape) == 1:
        for t in range(1, len(preds)):
            if np.sign(actuals[t] - actuals[t-1]) == np.sign(preds[t] - preds[t-1]):
                correct += 1
            total += 1
    # If 2D (multi-horizon batches)
    else:
        for i in range(preds.shape[0]):
            for t in range(1, preds.shape[1]):
                if np.sign(actuals[i, t] - actuals[i, t-1]) == np.sign(preds[i, t] - preds[i, t-1]):
                    correct += 1
                total += 1
                
    return correct / total if total > 0 else 0

def calculate_metrics(model_name, actuals, preds):
    """
    Consolidated function to calculate all metrics for the results table.
    """
    actuals_flat = actuals.flatten()
    preds_flat = preds.flatten()
    
    rmse = np.sqrt(mean_squared_error(actuals_flat, preds_flat))
    mae = mean_absolute_error(actuals_flat, preds_flat)
    direction = directional_accuracy_multistep(preds, actuals)
    
    return {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "Direction": direction
    }

def print_model_results(results_dict):
    """
    Prints model results in a clear, multi-line format with 4 decimal precision.
    """
    print(f"--- {results_dict['Model']} Results ---")
    print(f"RMSE:      {results_dict['RMSE']:.4f}")
    print(f"MAE:       {results_dict['MAE']:.4f}")
    print(f"Direction: {results_dict['Direction']:.4f}")
    


