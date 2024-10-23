import pandas as pd
import numpy as np
from helperfunctions import *

excel = 'past_roster.xlsx'

# Now we get all the dates

df = pd.read_excel(excel)

df['date'] = df['date'].dt.strftime('%d/%m/%Y')

dates = df.date.unique()


# Now we get all the stock in take for those dates

df1 = pd.read_excel('stock.xlsx')

# Simplify it down to the required columns

date = df1['Submitted at'].dt.strftime('%d/%m/%Y')
date = date.rename("date")
store = df1['Store'].map({'West End': 0, 'Indooroopilly': 1, 'Mitchelton': 2, 'Annerley': 3, 'Paddington': 4, 'Northcote': 5, 'Stones Corner': 6, 'Auchenflower': 7})
unsorted_bags = df1['Estimate bags-worth (1)']
sorted_bags = df1['Estimate bags-worth (2)']

df2 = pd.concat([date, store, unsorted_bags, sorted_bags], axis=1)

# Now we need to get the indices of the dates that allign with the stock_audits

dates2 = df2.date.unique()

intersect_dates = np.intersect1d(dates, dates2)

dates_frame = pd.DataFrame (intersect_dates)
filepath = 'dates.xlsx'
dates_frame.to_excel(filepath, index=False, header=False)


# uncomment to process data (Takes a few hours though)

'''
bags = bag_matrices(df2, intersect_dates)

df = pd.DataFrame(bags)
filepath = 'sorted_unsorted_bags.xlsx'
df.to_excel(filepath, index=False)
'''

'''
# Now with the dates we calculate the amount of workers that were on

length = len(intersect_dates)

roster = []

for i in range(length):
    x = req_trainer(excel, intersect_dates[i])
    roster_for_day = np.vstack((np.array(x[0]), np.array(x[1])))
    roster.append(roster_for_day)
    print("Progress:",i,'/',length)

roster = np.vstack(roster)

df = pd.DataFrame (roster)
filepath = 'historical_roster.xlsx'
df.to_excel(filepath, index=False)
'''