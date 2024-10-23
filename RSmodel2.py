import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helperfunctions import *
import torch
from torch import nn
from imblearn.over_sampling import ADASYN

# We want to take the transformed data and get a vector in R^4
# (workers on shift 1, workers on shift 2, Stock sorted, Stock Unsorted)^T

# Import the dates, stock and worker data
dates = pd.read_excel('dates.xlsx').to_numpy().reshape(-1)
roster = pd.read_excel('historical_roster.xlsx').to_numpy() # every two rows represents the number of workers on shift 1 and shift 2 over the 8 stores respectively
stock = pd.read_excel('sorted_unsorted_bags.xlsx').to_numpy() # every two rows represents the number of sorted and unsorted bags respectively

# Loop through each date and get the eight vectors for each
# If the vector is all zeros discard it

data = []

k = 0

for i in range(0, 2 * len(dates), 2):
    date = dates[k]
    r1 = roster[i]
    r2 = roster[i+1]
    r = r1 + r2
    s1 = stock[i]
    s2 = stock[i+1]
    for j in range(8):
        vec = np.array([r[j], s1[j], s2[j]])
        if np.sum(vec) != 0:
            data.append(vec)
    k += 1

data = np.array(data)

# remove 20 and 21 as just one sample (breaks ADASYN algorithm)

indices = []

for i in range(len(data[:,[0]].reshape(-1))):
    if data[:,[0]].reshape(-1)[i] > 19:
        indices.append(i)

data = np.delete(data, (indices), axis=0)

X_train = data[:,[1,2]]
y_train = data[:,[0]].reshape(-1).astype(int)

ada = ADASYN(random_state=42, n_neighbors=2)

X_train, y_train = ada.fit_resample(X_train, y_train)

X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()

# MLP

class MLP(nn.Module):
  
  #  Multilayer Perceptron.
  
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(2, 4),
      nn.SELU(),
      nn.Linear(4, 8),
      nn.SELU(),
      nn.Linear(8, 16),
      nn.SELU(),
      nn.Linear(16, 8),
      nn.SELU(),
      nn.Linear(8, 4),
      nn.Linear(4, 1)
    )


  def forward(self, x):
    # Forward pass
    return torch.flatten(self.layers(x))

# Initialize the MLP
mlp = MLP()

# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)


if __name__ == "__main__":
    # Run the training loop

    # 5 epochs at maximum
    epochs = 10000

    for epoch in range(0, epochs): 
    
        # Print epoch
        if epoch % 100 == 0:
            print("Epoch:", epoch+1, '/', end=' ')
        
        # Set current loss value
        current_loss = 0.0
        
        # Iterate over the DataLoader for training data

        # Get inputs
        inputs, targets = X_train, y_train
            
        # Zero the gradients
        optimizer.zero_grad()
            
        # Perform forward pass
        outputs = mlp(inputs)
            
        # Compute loss
        loss = loss_function(outputs, targets)
            
        # Perform backward pass
        loss.backward()
            
        # Perform optimization
        optimizer.step()
            
        # Print results
        current_loss += loss.item()
        
        if epoch % 100 == 0:
            print("Training Loss:", current_loss)
    

    # Process is complete.
    print('Training process has finished.')

    print(X_train[500:510])
    out = mlp(X_train[500:510])
    print(y_train[500:510])
    print(out)

    torch.save(mlp.state_dict(), 'model.pt')