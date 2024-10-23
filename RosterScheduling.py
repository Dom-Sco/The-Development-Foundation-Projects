import numpy as np 
import pandas as pd
from collections import Counter
from pulp import *
from helperfunctions import *
from RSmodel2 import *
import torch


# Calculate how many volunteers per store and if a trainer is required

date = '3/10/2024'

req = req_trainer('past_roster.xlsx', date)

shift_1_workers = req[0]
shift_2_workers = req[1]
trainer_needed = req[2]


# Other Parameters of the model and final data
# engineering for input into the cost function

# t # not sure what the costs are so will just set it all to random integers between 60 and 80

t = np.random.randint(low=60, high=80, size = I*J*K)


# r # done

r = []

for i in range(I):
    for j in range(J):
        if j == 0:
            r.append(trainer_needed.Shift_1.to_list())
        else:
            r.append(trainer_needed.Shift_2.to_list())

r = np.array(r).flatten()


# s

df = pd.read_excel('stock.xlsx')
df.rename(columns = {'Submitted at':'date'}, inplace = True)
stock = bag_matrix(df, date)

# w


s1 = np.array(shift_1_workers)
s2 = np.array(shift_2_workers)
workers = np.hstack((s1,s2))

w = []

for k in range(K):
    w.append(workers)

w = np.hstack((w))


# Calculate delta

# Get stock_audit for all stores and workers

stocks = []

for i in range(K):
    stocks.append(stock[:, i])

stocks = np.array(stocks)

# Get the assigned workers

workers_assigned = []

for i in range(K):
    x1 = s1[i]
    x2 = s2[i]

    vec = np.array([x1, x2])
    workers_assigned.append(vec)

workers_assigned = np.array(workers_assigned)

# Import model and calculate delta

model = MLP()
model.load_state_dict(torch.load('model.pt', weights_only=True))
model.eval()

model_input = torch.from_numpy(stocks).float()

with torch.no_grad():
    output = model(model_input)

delta = Delta(workers_assigned, output)

deltas = []

for i in range(J*I):
    deltas.append(delta)

delta = np.array(deltas).flatten()

# l
l = (delta < 100).astype(int)

# cost

c = cost(t, r, l, delta).tolist()


# Now with the cost function defined and calculated
# take the cost vector and use it in the linear model
# that will be created and optimised using the PULP library

prob = LpProblem("mincost", LpMinimize)


# First we define a tuple with the variable names
variables = []

for i in range(J*K*I):
    variables.append(str(i))

# Then we create a variable from a dictionary, using the variable names as keys
X = LpVariable.dicts("x", variables, lowBound = 0, upBound = 1, cat='Integer')


# Then we add the objective function to the model like
prob += (
    pulp.lpSum([
        c[i] * X[variables[i]]
        for i in range(len(X))])
)


# Then we add the constraints to the model

variables_new = []
for i in range(0, I*J*K, 8):
    variables_new.append(variables[i:i+8])

for i in range(I*J):
    prob += (
        pulp.lpSum([
            X[variables_new[i][k]]
            for k in range(K)])
    ) <= 1

# solve the problem
prob.solve()
LpStatus[prob.status]

# Display sols

for i in range(len(variables)):
    print(X[str(i)].varValue)