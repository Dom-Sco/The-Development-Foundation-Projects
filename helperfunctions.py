import pandas as pd
import numpy as np
from collections import Counter

# Number of stores, shifts and trainers

K = 8
J = 2
I = 15

def process_roster_for_a_date(excel_file_name, date): # both excel_file_name and date are strings
    # Import and process data
    df = pd.read_excel(excel_file_name)
    df = df[df['date'] == date]

    df['store'] = df['store'].map({'West End': 0, 'Indooroopilly': 1, 'Mitchelton': 2, 'Annerley': 3, 'Paddington': 4, 'Northcote': 5, 'Stones Corner': 6, 'Auchenflower': 7})

    data = np.array(df.values)
    return data

def volunteers_per_store(excel_file_name, date):
    data = process_roster_for_a_date(excel_file_name, date)

    stores = data[:, 1].tolist()
    counter = Counter(stores)
    counts = list(counter.items())

    volunteers = []

    for i in range(K):
        try:
            if counts[i][0] == i:
                volunteers.append(counts[i][1])
            else:
                volunteers.append(0)
        except IndexError:
            volunteers.append(0)

    return volunteers


# Calculate if the volunteers on require a trainer

def req_trainer(excel_file_name, date):
    data = process_roster_for_a_date(excel_file_name, date)

    period = data[:, 3]

    shift_1_workers = []
    shift_2_workers = []

    start = 0

    volunteers = volunteers_per_store(excel_file_name, date)

    for i in range(len(volunteers)):
        end = volunteers[i] + start
        cts = list(Counter(period[start:end]).items())
        if cts != []:
            if len(cts) == 2:
                if cts[0][0] == 0:
                    shift_1_workers.append(cts[0][1])
                    shift_2_workers.append(cts[1][1])
                else:
                    shift_1_workers.append(cts[1][1])
                    shift_2_workers.append(cts[0][1])
            else:
                if cts[0][0] == 0:
                    shift_1_workers.append(cts[0][1])
                    shift_2_workers.append(0)
                else:                
                    shift_1_workers.append(0)
                    shift_2_workers.append(cts[0][1])
        else:
            shift_1_workers.append(0)
            shift_2_workers.append(0)
        start = end

    # Now calculating if a trainer is required for each shift at each store

    dic = {'Shift_1': [0, 0, 0, 0, 0, 0, 0, 0], 'Shift_2': [0, 0, 0, 0, 0, 0, 0, 0]}
    trainer_needed = pd.DataFrame(dic)

    stores = data[:, 1]
    req_trainer = data[:, 6]

    start = 0

    for i in range(K):
        end = start + shift_1_workers[i]
        if sum(req_trainer[start:end]) > 0:
            trainer_needed.loc[i, ['Shift_1']] = [0]
        else:
            trainer_needed.loc[i, ['Shift_1']] = [1]
        start = end
        end = end + shift_2_workers[i]
        if sum(req_trainer[start:end]) > 0:
            trainer_needed.loc[i, ['Shift_2']] = [0]
        else:
            trainer_needed.loc[i, ['Shift_2']] = [1]
        start = end
    
    return [shift_1_workers, shift_2_workers, trainer_needed]



# Helper functions for stock audit data

# Want a function that given a date will return a matrix of bags_worth (top is 1 and bottom is 2)

def bag_matrix(df, date):
    submissions = df.loc[df['date'] == date]
    vectors = []
    
    for i in range(K):
        if i in submissions.Store.values:
            row = submissions.loc[df['Store'] == i]
            x1 = row['Estimate bags-worth (1)']
            x2 = row['Estimate bags-worth (2)']
            vectors.append(np.array([x1,x2]))
        else:
            vectors.append(np.array([[0],[0]]))
    
    estimates = np.hstack(vectors)

    # Adding this cause the data seems to have a formatting error somewhere I can't work out, either that or a bug in pandas
    if estimates.shape[1] == 9:
        estimates = np.delete(estimates, -1, axis=1)

    return estimates

# want a function that takes a list of dates and a dataframe and will return an array of all the
# bag_matrices

def bag_matrices(df, dates):
    length = len(dates)
    matrices = []

    for i in range(length):
        if dates[i] in df.date.values:
            matrices.append(bag_matrix(df, dates[i]))
    estimates = np.vstack(matrices)
    return estimates



# Define function to calculate delta

def Delta(w, out):
    out = out.numpy()
    w = np.sum(w, axis = 1)

    delta = (w / out) * 100
    return delta


# Cost Function

# Takes as input vectors for trainer cost for shift (t),
# if trainer is required for shift (r),
# output of linear classifier (l)

def cost(t, r, l, delta):
    cost = (1 - r) * t + (-1)**(l) * delta
    return cost