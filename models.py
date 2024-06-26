from torch.nn import Transformer
from torch import nn
import torch
import math
import torch.nn.functional as F
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
import torch.nn.functional as F





class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=4, output_size=1, activation_name='None', num_heads=None, dropout=None):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation_name = activation_name
        if self.activation_name != 'None':
            self.activation = getattr(nn, activation_name)()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        if self.activation_name != 'None':
            out = self.activation(out)
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=4, output_size=1,activation_name='GELU', num_heads=None, dropout=None):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=4, output_size=1,activation_name='GELU',  num_heads=None, dropout=None):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out




class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout=0.0):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2 * dilation, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2 * dilation, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, input_size=1, num_channels=[32, 64], kernel_size=7, dropout=0.1, activation_name='GELU'):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  
        out = self.network(x)
        out = out[:, :, -1]  
        out = self.fc(out)
        out = out.unsqueeze(-1)  
        return out



class MLPModel(nn.Module):
    def __init__(self, input_size=1, layer_sizes=[64, 64, 32, 32, 16, 16], dropout=0.1, activation_name='GELU'):
        super(MLPModel, self).__init__()
        self.flatten = nn.Flatten()
        layers = []
        
        layers.append(nn.Linear(50 * input_size, layer_sizes[0]))
        layers.append(nn.ReLU())

        for i in range(1, len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(layer_sizes[-1], 1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        out = self.network(x)
        out = out.unsqueeze(-1).unsqueeze(-1) 
        return out







#### Transformer
class TransformerModel(nn.Module):
    def __init__(self, iw=50, d_model=512, num_heads=8, num_layers=4, input_size=1, dropout=0.1, ow=1):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers) 
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder = nn.Sequential(
            nn.Linear(1, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, d_model)
            )
        
        self.linear =  nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
            )

        self.linear2 = nn.Sequential(
            nn.Linear(iw, (iw+ow)//2),
            nn.ReLU(),
            nn.Linear((iw+ow)//2, ow)
            ) 

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        srcmask = self.generate_square_subsequent_mask(src.shape[1]).to('cuda')
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src.transpose(0,1), srcmask).transpose(0,1)
        output = self.linear(output)[:,:,0]
        output = self.linear2(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def gen_attention_mask(x):
    mask = torch.eq(x, 0)
    return mask

#######
#######VAE

### Basic Layer
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


### VAE ENCODER DECODER
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_mean = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        self.swish = Swish()

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        h = self.swish(h)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_dim, latent_dim, output_window):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_out = nn.Linear(hidden_dim * 2, input_size)
        self.output_window = output_window
        self.swish = Swish()

    def forward(self, z):
        z = z.unsqueeze(1).repeat(1, self.output_window, 1)
        output, _ = self.lstm(z)
        output = self.swish(output)
        return self.fc_out(output)


### LSTM-HVAE 구현
class TimeHVAE(nn.Module):
    def __init__(self, input_size=1, hidden_dim=128, latent_dim=20, output_window=1, dropout=0):
        super(TimeHVAE, self).__init__()
        self.encoder = Encoder(input_size, hidden_dim, latent_dim)
        self.res_encoder = Encoder(input_size + latent_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(input_size, hidden_dim, latent_dim, output_window)
        self.fc_delta_z2_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_delta_z2_logvar = nn.Linear(latent_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        z1_mu, z1_logvar = self.encoder(x)
        z1 = self.reparameterize(z1_mu, z1_logvar)

        x_with_z1 = torch.cat([x, z1.unsqueeze(1).repeat(1, x.size(1), 1)], dim=2)
        z2_mu, z2_logvar = self.res_encoder(x_with_z1)

        z2_mu_res = self.fc_delta_z2_mu(z1)
        z2_logvar_res = self.fc_delta_z2_logvar(z1)

        z2_mu = z2_mu + z2_mu_res
        z2_logvar = z2_logvar + z2_logvar_res

        z2 = self.reparameterize(z2_mu, z2_logvar)
        recon_x = self.decoder(z2)

        return recon_x, z1_mu, z1_logvar, z2_mu, z2_logvar

    def loss_function(self, x, recon_x, z1_mu, z1_logvar, z2_mu, z2_logvar):
        recon_loss = nn.MSELoss()(recon_x, x)
        kl_loss_z1 = -0.5 * torch.sum(1 + z1_logvar - z1_mu.pow(2) - z1_logvar.exp())
        kl_loss_z2 = -0.5 * torch.sum(1 + z2_logvar - z2_mu.pow(2) - z2_logvar.exp())
        return recon_loss + kl_loss_z1 + kl_loss_z2

    def sample(self, n):
        with torch.no_grad():
            z1 = torch.randn(n, self.encoder.fc_mean.out_features).to(next(self.parameters()).device)
            z2_mu_res = self.fc_delta_z2_mu(z1)
            z2_logvar_res = self.fc_delta_z2_logvar(z1)
            z2 = self.reparameterize(z2_mu_res, z2_logvar_res)
            samples = self.decoder(z2)
            return samples



##### ARIMA
class ARIMAModel:
    def __init__(self, order=(1, 0, 1)):
        self.order = order

    def fit(self, series_data):
        dates = pd.date_range(start='2022-01-01', periods=len(series_data), freq='D')
        series = pd.Series(series_data, index=dates)
        
        # ARIMA 모델 생성 및 훈련
        model = ARIMA(series, order=self.order)
        model_fit = model.fit()
        return model_fit

    def forecast(self, model_fit):
        forecast = model_fit.forecast(steps=1)
        return forecast[0]
