import xgboost as xgb
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

class Model(torch.nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual
        
        self.models = []
        self.scalers = []
        
        for _ in range(self.channels):
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
            self.models.append(model)
            self.scalers.append(StandardScaler())

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        batch_size, _, _ = x.shape
        output = torch.zeros((batch_size, self.pred_len, self.channels), dtype=torch.float32)
        
        for i in range(self.channels):
            channel_data = x[:, :, i]
            
            # Reshape data for XGBoost
            X = channel_data.reshape(-1, self.seq_len)
            X_scaled = self.scalers[i].fit_transform(X)
            
            # Train the model
            self.models[i].fit(X_scaled, np.zeros(X_scaled.shape[0]))  # Dummy y values
            
            # Make predictions
            X_future = np.zeros((batch_size, self.seq_len))
            X_future[:, -self.seq_len:] = channel_data[:, -self.seq_len:]
            X_future_scaled = self.scalers[i].transform(X_future)
            predictions = self.models[i].predict(X_future_scaled)
            
            output[:, :, i] = torch.from_numpy(predictions.reshape(-1, self.pred_len)).float()
        
        return output

    def test(self, x):
        return self.forward(x)