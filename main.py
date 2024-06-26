import argparse
import torch
import torch.optim as optim
from data import download_data, preprocess_data
from models import LSTMModel, GRUModel, TransformerModel, TimeHVAE, ARIMAModel, RNNModel, TCNModel, MLPModel
from train import create_dataloader, train_model, evaluate_model, vae_train_model, vae_evaluate_model, arima_train_model, arima_evaluate_model
from utils import plot_predictions, compare_prediction, compare_evalutation, indivi_evaluation
import numpy as np

def run_experiment(model_name, ticker, model_params, training_params, device, optimizer_name, optimizer_params):
    data = download_data(ticker, '2022-01-01', '2024-01-01')
    data = data[['Close']]
    X, Y, scaler = preprocess_data(data, training_params['time_step'], model_params['input_size'])
    
    if model_name == 'lstm':
        model = LSTMModel(**model_params).to(device)
    elif model_name == 'gru':
        model = GRUModel(**model_params).to(device)
    elif model_name == 'rnn':
        model = RNNModel(**model_params).to(device)
    elif model_name == 'tcn':
        model = TCNModel(**model_params).to(device)
    elif model_name == 'mlp':
        model = MLPModel(**model_params).to(device)
    elif model_name == 'transformer':
        model = TransformerModel(**model_params).to(device)
    elif model_name == 'timevae':
        model = TimeHVAE(**model_params).to(device)
    elif model_name == 'arima':
        model = ARIMAModel()
    else:
        raise ValueError("Invalid model name. Choose from 'lstm', 'gru', 'transformer'.")
    
    train_loader = create_dataloader(X, Y, training_params['batch_size'])
    
    criterion = torch.nn.MSELoss()
    if model_name == 'arima':
        None
    else:
        optimizer = getattr(optim, optimizer_name)(model.parameters(), **optimizer_params)
    
    if model_name == 'timevae':
        vae_train_model(model, train_loader, criterion, optimizer, training_params['num_epochs'], device)
        predictions, actuals = vae_evaluate_model(model, train_loader, device)
        score = 0
    
    elif model_name == 'arima':
        trained_models = arima_train_model(model, train_loader)
        predictions, actuals = arima_evaluate_model(model, train_loader, trained_models)
        score = 0
    
    else:
        train_model(model, train_loader, criterion, optimizer, training_params['num_epochs'], device)
        predictions, actuals = evaluate_model(model, train_loader, device)
    
        # Compute the score
        score = criterion(model(torch.tensor(X, dtype=torch.float32).to(device)), torch.tensor(Y, dtype=torch.float32).view(-1, 1).to(device)).item()
        
    return predictions, actuals, scaler, score




def get_optimizer_params(args):
    optimizer_params = {}
    if args.optimizer in ['Adam', 'AdamW', 'Adamax']:
        optimizer_params = {'lr': args.learning_rate, 'betas': (args.beta1, args.beta2), 'eps': args.eps, 'weight_decay': args.weight_decay}
    elif args.optimizer in ['SGD', 'RMSprop']:
        optimizer_params = {'lr': args.learning_rate, 'momentum': args.momentum}
    elif args.optimizer == 'Adagrad':
        optimizer_params = {'lr': args.learning_rate, 'lr_decay': args.lr_decay, 'weight_decay': args.weight_decay}
    elif args.optimizer == 'ASGD':
        optimizer_params = {'lr': args.learning_rate, 'lambd': args.lambd, 'alpha': args.alpha, 't0': args.t0, 'weight_decay': args.weight_decay}
    elif args.optimizer == 'LBFGS':
        optimizer_params = {'lr': args.learning_rate, 'max_iter': args.max_iter, 'max_eval': args.max_eval, 'tolerance_grad': args.tolerance_grad, 'tolerance_change': args.tolerance_change, 'history_size': args.history_size}
    elif args.optimizer == 'Rprop':
        optimizer_params = {'lr': args.learning_rate, 'etas': (args.eta1, args.eta2), 'step_sizes': (args.step_size_min, args.step_size_max)}
    return optimizer_params





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Price Prediction using various models and optimizers")
    parser.add_argument('--model', type=str, required=True, choices=['lstm', 'gru', 'rnn', 'tcn', 'mlp', 'transformer', 'timevae', 'arima'], help='Model type to use for prediction')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    
    ### Train model parser
    parser.add_argument('--input_size', type=int, required=True, help='Input size for the model')
    parser.add_argument('--hidden_size', type=int, default=32, help='Hidden size for LSTM/GRU')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers for LSTM/GRU/Transformer')
    parser.add_argument('--activation', type=str, default='None', choices=['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU', 'ELU', 'SELU', 'Softplus', 'GELU'], help='Activation function to use in the model')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for Transformer')
    ###

    parser.add_argument('--time_step', type=int, default=100, help='Time step for the input sequences')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training')
    
    ### optimizer parser
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD', 'RMSprop', 'AdamW', 'Adagrad', 'Adamax', 'ASGD', 'LBFGS', 'Rprop'], help='Optimizer to use for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizers like SGD and RMSprop')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for optimizers like Adam and AdamW')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for optimizers like Adam and AdamW')
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon for optimizers like Adam and AdamW')
    parser.add_argument('--lr_decay', type=float, default=0, help='Learning rate decay for Adagrad')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimizers like Adam and AdamW')
    parser.add_argument('--lambd', type=float, default=1e-4, help='Decay term for ASGD')
    parser.add_argument('--alpha', type=float, default=0.75, help='Power for ASGD')
    parser.add_argument('--t0', type=float, default=1e6, help='Start point for averaging in ASGD')
    parser.add_argument('--max_iter', type=int, default=20, help='Maximum number of iterations for LBFGS')
    parser.add_argument('--max_eval', type=int, default=None, help='Maximum number of function evaluations for LBFGS')
    parser.add_argument('--tolerance_grad', type=float, default=1e-7, help='Gradient tolerance for LBFGS')
    parser.add_argument('--tolerance_change', type=float, default=1e-9, help='Change tolerance for LBFGS')
    parser.add_argument('--history_size', type=int, default=100, help='History size for LBFGS')
    parser.add_argument('--eta1', type=float, default=0.5, help='Eta1 for Rprop')
    parser.add_argument('--eta2', type=float, default=1.2, help='Eta2 for Rprop')
    parser.add_argument('--step_size_min', type=float, default=1e-6, help='Minimum step size for Rprop')
    parser.add_argument('--step_size_max', type=float, default=50, help='Maximum step size for Rprop')
    ###
    
    parser.add_argument('--file_name', type=str, required=True, help='File name for save')
    parser.add_argument('--task_name', type=str, default='None', choices=['optimizer', 'activation', 'num_layer', 'num_node'], help='Model type to use for prediction')
    parser.add_argument('--compare_model', type=bool, default=False, help='compare experiments whole model')


    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_params = {
        'input_size': args.input_size,
        'hidden_size': args.hidden_size if args.model == 'lstm' else None,
        'num_layers': args.num_layers if args.model == 'lstm'else None,
        'activation_name' : args.activation,
        'dropout': args.dropout
    }
    
    training_params = {
        'time_step': args.time_step,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs
    }
    
    optimizer_params = {}
    if args.optimizer in ['Adam', 'AdamW', 'Adamax']:
        optimizer_params = {'lr': args.learning_rate, 'betas': (args.beta1, args.beta2), 'eps': args.eps, 'weight_decay': args.weight_decay}
    elif args.optimizer in ['SGD', 'RMSprop']:
        optimizer_params = {'lr': args.learning_rate, 'momentum': args.momentum}
    elif args.optimizer == 'Adagrad':
        optimizer_params = {'lr': args.learning_rate, 'lr_decay': args.lr_decay, 'weight_decay': args.weight_decay}
    elif args.optimizer == 'ASGD':
        optimizer_params = {'lr': args.learning_rate, 'lambd': args.lambd, 'alpha': args.alpha, 't0': args.t0, 'weight_decay': args.weight_decay}
    elif args.optimizer == 'LBFGS':
        optimizer_params = {'lr': args.learning_rate, 'max_iter': args.max_iter, 'max_eval': args.max_eval, 'tolerance_grad': args.tolerance_grad, 'tolerance_change': args.tolerance_change, 'history_size': args.history_size}
    elif args.optimizer == 'Rprop':
        optimizer_params = {'lr': args.learning_rate, 'etas': (args.eta1, args.eta2), 'step_sizes': (args.step_size_min, args.step_size_max)}

    opimizer_list = ['Adam',  'RMSprop', 'AdamW', 'Adagrad', 'Adamax', 'Rprop']
    activation_list= ['ReLU', 'Sigmoid', 'Tanh', 'LeakyReLU', 'ELU', 'SELU', 'Softplus', 'GELU']
    num_layers_list = [1,2,3,4,5,6,7,8]
    hidden_size_list = [8,16,24,32,40,48,56,64]

    if args.compare_model == True:
        if args.task_name =='optimizer':
            val_list = opimizer_list
        elif args.task_name ==  'activation':
            val_list = activation_list
        elif args.task_name ==  'num_layer':
            val_list = num_layers_list
        elif args.task_name ==  'num_node':
            val_list = hidden_size_list
        
        whole_predictions = []
        whole_actuals = []
        whole_scalers = []

        for val in val_list:
            if args.task_name =='optimizer':
                args.optimizer = val
                optimizer_params = get_optimizer_params(args)
            elif args.task_name =='activation':
                model_params['activation_name'] = val
            elif args.task_name =='num_layer':
                model_params['num_layers'] = val
            elif args.task_name =='num_node':
                model_params['hidden_size'] = val
            
            
            print(model_params)
            predictions, actuals, scaler, score = run_experiment(args.model, args.ticker, model_params, training_params, device, args.optimizer, optimizer_params)
            whole_predictions.append(predictions)
            whole_actuals.append(actuals)
            whole_scalers.append(scaler)


        compare_prediction(val_list, whole_predictions, whole_actuals, whole_scalers, args.task_name, str(val))
        compare_evalutation(val_list, whole_predictions, whole_actuals, whole_scalers, args.task_name, str(val))


    else:
        predictions, actuals, scaler, score = run_experiment(args.model, args.ticker, model_params, training_params, device, args.optimizer, optimizer_params)
            
        plot_predictions(predictions, actuals, scaler, args.file_name)
        indivi_evaluation(predictions, actuals, scaler, args.model,  args.file_name)
    
