import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def create_dataloader(X, Y, batch_size=64, train_ratio=0.8):
    train_size = int(len(X) * train_ratio)
    
    X_train, X_val = X[:train_size], X[train_size:]
    Y_train, Y_val = Y[:train_size], Y[train_size:]
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))
    
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader, val_loader


def train_model(model, dataloader, criterion, optimizer, num_epochs=20, device='cpu'):
    model.train()
    for epoch in range(num_epochs):
        for X_batch, Y_batch in dataloader[0]:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            outputs = model(X_batch)

            Y_batch = Y_batch.view(-1, 1)

            loss = criterion(outputs, Y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def evaluate_model(model, dataloader, device='cpu'):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, Y_batch in dataloader[1]:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)
            predictions.append(outputs.cpu().numpy())
            actuals.append(Y_batch.cpu().numpy().reshape(-1, 1))
    return np.concatenate(predictions), np.concatenate(actuals)



def vae_train_model(model, dataloader, criterion, optimizer, num_epochs=20,  device='cuda'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader[0]):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            recon_batch, mean, logvar, z2_mu, z2_var = model(data)
            loss = model.loss_function(target, recon_batch, mean, logvar, z2_mu, z2_var)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        
        print(f'Epoch {epoch+1}, Train Loss: {loss.item()}')
    
    return model


def vae_evaluate_model(model, test_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    

    test_loss = 0
    predictions, actuals = [], []

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader[1]):
            data, target = data.to(device), target.to(device)
            recon, mean, logvar, z2_mu, z2_var = model(data)
            loss = model.loss_function(target, recon, mean, logvar, z2_mu, z2_var)
            test_loss += loss.item()

            recon = recon.squeeze(2)

            predictions.append(recon.cpu().numpy())
            actuals.append(target.cpu().numpy().reshape(-1, 1))

    print(f'Test loss: {loss.item()}')

    return np.concatenate(predictions), np.concatenate(actuals)

def arima_train_model(model, dataloader):
    trained_models = []
    for X_batch, Y_batch in dataloader[0]:
        X_batch_np = X_batch.numpy()
        for i in range(X_batch_np.shape[0]):
            series_data = X_batch_np[i, :, 0]
            model_fit = model.fit(series_data)
            trained_models.append(model_fit)
    return trained_models

def arima_evaluate_model(model, dataloader, trained_models):
    predictions, actuals = [], []
    model_index = 0
    for X_batch, Y_batch in dataloader[1]:
        X_batch_np = X_batch.numpy()
        Y_batch_np = Y_batch.numpy()
        batch_predictions = []
        
        for i in range(X_batch_np.shape[0]):
            model_fit = trained_models[model_index]
            model_index += 1
            prediction = model.forecast(model_fit)
            batch_predictions.append(prediction)
        
        batch_predictions = np.array(batch_predictions).reshape(-1, 1, 1)
        predictions.append(batch_predictions)
        actuals.append(Y_batch_np)

    return np.concatenate(predictions, axis=0), np.concatenate(actuals, axis=0)
