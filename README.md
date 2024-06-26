# README for Task2_OAI(Optimization for AI) with PyTorch

## Overview

This project aims to predict stock prices using various Deep learning models and optimizers. The main script, `main.py`, allows users to select a model, optimizer, and other training parameters to train and evaluate the model on historical stock price data.

## Requirements

-   Python 3.6 or higher
-   PyTorch
-   Numpy
-   Pandas
-   Matplotlib
-   argparse

## Installation

1. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The `main.py` script can be run from the command line with various arguments to specify the model, optimizer, and other parameters. Below is a detailed description of the arguments:

### Arguments

-   `--model`: The type of model to use for prediction. Choices are `lstm`, `gru`, `rnn`, `tcn`, `mlp`, `transformer`, `timevae`, `arima`.
-   `--ticker`: The stock ticker symbol (default: `AAPL`).
-   `--input_size`: Input size for the model (required).
-   `--hidden_size`: Hidden size for LSTM/GRU (default: 32).
-   `--num_layers`: Number of layers for LSTM/GRU/Transformer (default: 2).
-   `--activation`: Activation function to use in the model. Choices are `ReLU`, `Sigmoid`, `Tanh`, `LeakyReLU`, `ELU`, `SELU`, `Softplus`, `GELU`.
-   `--dropout`: Dropout rate for Transformer (default: 0.1).
-   `--time_step`: Time step for the input sequences (default: 100).
-   `--batch_size`: Batch size for training (default: 64).
-   `--num_epochs`: Number of epochs for training (default: 20).
-   `--optimizer`: Optimizer to use for training. Choices are `Adam`, `SGD`, `RMSprop`, `AdamW`, `Adagrad`, `Adamax`, `ASGD`, `LBFGS`, `Rprop`.
-   `--learning_rate`: Learning rate for the optimizer (default: 0.001).
-   `--momentum`: Momentum for optimizers like SGD and RMSprop (default: 0.9).
-   `--beta1`: Beta1 for optimizers like Adam and AdamW (default: 0.9).
-   `--beta2`: Beta2 for optimizers like Adam and AdamW (default: 0.999).
-   `--eps`: Epsilon for optimizers like Adam and AdamW (default: 1e-8).
-   `--lr_decay`: Learning rate decay for Adagrad (default: 0).
-   `--weight_decay`: Weight decay for optimizers like Adam and AdamW (default: 0).
-   `--lambd`: Decay term for ASGD (default: 1e-4).
-   `--alpha`: Power for ASGD (default: 0.75).
-   `--t0`: Start point for averaging in ASGD (default: 1e6).
-   `--max_iter`: Maximum number of iterations for LBFGS (default: 20).
-   `--max_eval`: Maximum number of function evaluations for LBFGS (default: None).
-   `--tolerance_grad`: Gradient tolerance for LBFGS (default: 1e-7).
-   `--tolerance_change`: Change tolerance for LBFGS (default: 1e-9).
-   `--history_size`: History size for LBFGS (default: 100).
-   `--eta1`: Eta1 for Rprop (default: 0.5).
-   `--eta2`: Eta2 for Rprop (default: 1.2).
-   `--step_size_min`: Minimum step size for Rprop (default: 1e-6).
-   `--step_size_max`: Maximum step size for Rprop (default: 50).
-   `--file_name`: File name for saving results (required).
-   `--task_name`: Task name for comparing models. Choices are `optimizer`, `activation`, `num_layer`, `num_node`.
-   `--compare_model`: Boolean flag to indicate if comparing multiple models (default: False).

### Example Usage

1. Running a single model:

    ```bash
    python main.py --model lstm --ticker ^SPX --input_size 1 --time_step 50 --batch_size 64 --num_epochs 100 --file_name lstm
    ```

2. Comparing multiple optimizers:
    ```bash
    python main.py --model lstm --ticker ^SPX --input_size 1 --time_step 50 --batch_size 64 --num_epochs 100 --compare_model True --task_name activation --file_name temp
    ```

## Code Structure

-   `main.py`: The main script for running experiments.
-   `data.py`: Contains functions for downloading and preprocessing data.
-   `models.py`: Contains definitions for various models (LSTM, GRU, Transformer, etc.).
-   `train.py`: Contains functions for training and evaluating models.
-   `utils.py`: Contains utility functions for plotting and comparing predictions.

## Function Descriptions

### `run_experiment`

Runs the experiment for a given model and optimizer.

**Parameters:**

-   `model_name`: The name of the model to use.
-   `ticker`: The stock ticker symbol.
-   `model_params`: Parameters for the model.
-   `training_params`: Training parameters.
-   `device`: The device to use for training (CPU or GPU).
-   `optimizer_name`: The name of the optimizer to use.
-   `optimizer_params`: Parameters for the optimizer.

**Returns:**

-   `predictions`: Predicted values.
-   `actuals`: Actual values.
-   `scaler`: Scaler used for data normalization.
-   `score`: Evaluation score of the model.

### `get_optimizer_params`

Gets the optimizer parameters based on the selected optimizer.

**Parameters:**

-   `args`: The parsed command line arguments.

**Returns:**

-   `optimizer_params`: Dictionary of optimizer parameters.

## Results and Evaluation

The results of the predictions are saved as plots, and the performance of the models is evaluated using mean MSE, MAE, RMSE, MPE, $R^2$. Additional comparison plots and evaluations can be generated if the `--compare_model` flag is set.

## Contributing

Contributions are welcome!

## License

This project is licensed under the UNIST License.

---
