import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from helperfunctions import *


# Sorting data and dealing with missing values

df = pd.read_excel('product_sales_counts.xlsx')
df["date"] = df.groupby("date").ngroup()
df['store'] = df['store'].map({'West End': 0, 'Indooroopilly': 1, 'Mitchelton': 2, 'Annerley': 3, 'Paddington': 4, 'Northcote': 5, 'Stones Corner': 6})
    
n = len(pd.unique(df['date'])) # max len of data

dataframes = []

for i in range(K - 1): # No data for Auchenflower
    cur_data = df.loc[df['store'] == i]
    values = []
    dates = cur_data['date'].values
    for j in range(n):
        if j not in dates:
            values.append(j)
        
    m = len(values)
    values = np.array(values).reshape(m,1)
    store = i * np.ones(m).reshape(m,1)

    nans = np.empty((m,29,))
    nans[:] = np.nan

    df1 = np.hstack((np.hstack((values, store)), nans))

    cur_data = pd.DataFrame(np.vstack((cur_data.to_numpy(), df1))).sort_values(by = 0)
    dataframes.append(cur_data)
    
dataframes2 = []

for i in range(len(dataframes)):
    dataframe = dataframes[i].drop([0, 1], axis=1)
    dataframes2.append(dataframe.interpolate(method='linear', axis=0).dropna())

m = dataframes2[5].shape[0]

# Split it into training and testing sets

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature.values)
        y.append(target.values)
    return torch.tensor(X).float(), torch.tensor(y).float()


def train_test_split(i):
    if i == 5: # North cote is missing data at the start (linear interpolation unable to fix this)
        train_size = int(0.67 * m)
        test_size = m - train_size
    else:
        train_size = int(0.67 * n)
        test_size = n - train_size
    
    train, test = dataframes2[i][:train_size], dataframes2[i][train_size:]

    lookback = 7
    X_train, y_train = create_dataset(train, lookback=lookback)
    X_test, y_test = create_dataset(test, lookback=lookback)

    return X_train, y_train, X_test, y_test

# LSTM model


class SalesModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=29, hidden_size=100, num_layers=3, batch_first=True)
        self.linear = nn.Linear(100, 29)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


# Training function

def train(X_train, y_train, X_test, y_test, n_epochs, file_name):
    model = SalesModel()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)
    
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        if epoch % 100 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
    
    torch.save(model.state_dict(), file_name)

if __name__ == "__main__":

    stores = ['West End', 'Indooroopilly', 'Mitchelton', 'Annerley', 'Paddington', 'Northcote', 'Stones Corner']

    
    for i in range(K-1):
        X_train, y_train, X_test, y_test = train_test_split(i)
        
        print("Training on sales data for ", stores[i])
        file_name = "SalesModel" + stores[i] + ".pt"
        n_epochs = 2000
        train(X_train, y_train, X_test, y_test, n_epochs, file_name)