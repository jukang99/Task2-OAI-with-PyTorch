import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
import os
import pandas as pd

def plot_predictions(predictions, actuals, scaler, file_name):
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    actuals = scaler.inverse_transform(actuals.reshape(-1, 1))

    plt.figure(figsize=(16,8))
    plt.plot(actuals, label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.legend()
    os.makedirs('results', exist_ok=True)
    plt.savefig(os.path.join('results', file_name+'.png'))



def indivi_evaluation(predictions, actuals, scaler, model_name, file_name):
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    actuals = scaler.inverse_transform(actuals.reshape(-1, 1))
    mae, mse, rmse, mpe, r2 = calculate_metrics(actuals, predictions)

    eval_df = pd.DataFrame([[model_name, mae, mse, rmse, mpe, r2]], columns=['model_name', 'mae', 'mse', 'rmse', 'mpe', 'r2'])
    eval_df.to_csv(os.path.join('results', file_name+'.csv'))
    print(eval_df)

def compare_prediction(model_list, whole_predictions, whole_actuals, whole_scaler, task_name, file_name):
    plt.figure(figsize=(16,8))
    
    for model_idx in range(len(model_list)):
        predictions = whole_scaler[model_idx].inverse_transform(whole_predictions[model_idx].reshape(-1, 1))
        actuals = whole_scaler[model_idx].inverse_transform(whole_actuals[model_idx].reshape(-1, 1))
        if model_idx ==0:
            plt.plot(actuals, label=f'Actual Prices')
        plt.plot(predictions, linestyle='dashed', label=f'{model_list[model_idx]}')
    
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    save_path = f'results/{task_name}'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, task_name+'.png'))



def compare_evalutation(model_list, whole_predictions, whole_actuals, whole_scaler,task_name, file_name):
    
    eval_score = []
    for model_idx in range(len(model_list)):
        predictions = whole_scaler[model_idx].inverse_transform(whole_predictions[model_idx].reshape(-1, 1))
        actuals = whole_scaler[model_idx].inverse_transform(whole_actuals[model_idx].reshape(-1, 1))
        mae, mse, rmse, mpe, r2 = calculate_metrics(actuals, predictions)
        eval_score.append([model_list[model_idx], mae, mse, rmse, mpe, r2])
        
    
    save_path = f'results/{task_name}'
    os.makedirs(save_path, exist_ok=True)
    eval_df = pd.DataFrame(eval_score, columns=['model_name', 'mae', 'mse', 'rmse', 'mpe', 'r2'])
    eval_df.to_csv(os.path.join(save_path, task_name+'.csv'))
    print(eval_df)


def MPE(y_test, y_pred): 
	return np.mean((y_test - y_pred) / y_test) * 100

def calculate_metrics(true_values, predictions):
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mpe = MPE(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    return mae, mse, rmse, mpe, r2




