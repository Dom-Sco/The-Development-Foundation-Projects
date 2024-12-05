import pandas as pd
import numpy as np
from helperfunctions import *
import matplotlib.pyplot as plt
import matplotlib
from statsmodels.tsa.stattools import adfuller
from pandas import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
matplotlib.use('TkAgg')

def stock_data(df):
    df = df.iloc[1469:]
    df['Submitted at'] = df['Submitted at'].dt.strftime('%d/%m/%Y')
    df['Store'] = df['Store'].map({'West End': 0, 'Indooroopilly': 1, 'Mitchelton': 2, 'Annerley': 3, 'Paddington': 4, 'Northcote': 5, 'Stones Corner': 6, 'Auchenflower': 7})
    
    df1 = df[['Submitted at', 'Store', 'Estimate bags-worth (1)', 'Estimate bags-worth (2)']]
    
    return df1

# define a function that works out the total number of items sold at each store on a given day

def items_sold(df):
    df['date'] = df['date'].dt.strftime('%d/%m/%Y')
    df['store'] = df['store'].map({'West End': 0, 'Indooroopilly': 1, 'Mitchelton': 2, 'Annerley': 3, 'Paddington': 4, 'Northcote': 5, 'Stones Corner': 6, 'Auchenflower': 7})

    df1 = df[['date', 'store']]

    c = df.columns

    df1['items_sold'] = df.loc[:,c[-29:]].sum(1)
    
    return df1


# Define a function to work out current stock

# First use stock audit data and partition the sales data into the dates between stock audits and partition it for each store

# Calculate the number of workers rostered on each day for each store

def workers_per_store_per_day(df):
    # First seperate the dataframe into 8 dataframes for each store
    df['store'] = df['store'].map({'West End': 0, 'Indooroopilly': 1, 'Mitchelton': 2, 'Annerley': 3, 'Paddington': 4, 'Northcote': 5, 'Stones Corner': 6, 'Auchenflower': 7})

    stores = []

    for i in range(K-1):
        stores.append(df.loc[df['store'] == i])
    
    # Now get the counts for each unique date for each store

    counts = []

    for i in range(K-1):
        cts = stores[i]['date'].value_counts()
        cts_df = pd.DataFrame({'date':cts.index, 'count':cts.values})
        cts_df["date"] = pd.to_datetime(cts_df["date"]).dt.strftime('%d/%m/%Y')
        counts.append(cts_df.to_numpy())

    return counts


def partition(audit, sales):
    bags_store = []
    sales_store = []

    for i in range(K):
        bags_store.append(audit.loc[audit['Store'] == i].values)
        sales_store.append(sales.loc[sales['store'] == i].values)

    return bags_store, sales_store


def eng_data():
    dataframes = []

    df = pd.read_excel('stock.xlsx')
    df1 = stock_data(df)
    df_items = pd.read_excel('product_sales_counts.xlsx')
    sales = items_sold(df_items)
    
    bags_store, sales_store = partition(df1, sales)


    roster = pd.read_excel('past_roster.xlsx')
    roster = workers_per_store_per_day(roster)

    for i in range(K-1):
        d1 = pd.DataFrame(bags_store[i])
        d2 = pd.DataFrame(sales_store[i])

        r = pd.DataFrame(roster[i])
        r = r.sort_values(by=0)

        e1 = pd.merge(d1, d2, how='outer', on=0)
        e1 = pd.merge(e1, r, how='outer', on=0)
        e1[0] = pd.to_datetime(e1[0], dayfirst=True)
        e1 = e1.sort_values(by=0)
        e1 = e1.drop('1_x', axis=1)
        e1 = e1.drop('1_y', axis=1)
        dates = e1[0].to_list()
        
        e1 = e1.drop(0, axis=1)
        e1 = e1.apply(pd.to_numeric)
        e1 = e1.set_axis(dates)
        e1 = e1.interpolate(method='time', axis=0)
        e1 = e1.dropna(how='any')

        if e1.empty == False:
            dataframes.append(e1)
        else:
            print(i)
    return dataframes


def diff(df):
    diff_enough = False
    d = 0

    while diff_enough == False:
        result = adfuller(df)
        p = result[1]
        if p > 0.05:
            d += 1
            df = np.diff(df)
        else:
            diff_enough = True

    return df, d

if __name__ == "__main__":
    dataframes = eng_data()
    
    # Now we calculate the d parameters
    diffs = []
    d = []

    for i in range(K-2):
        series = np.round(dataframes[i][3].to_numpy())
        df, ds = diff(series)
        diffs.append(df)
        d.append(ds)
    
    d = np.array(d)

    # only uncomment if parameters need to be restablished
    '''
    # Now find p parameters
    print("p Parameters")
    for i in range(K-2):
        # PACF plot of 1st differenced series
        plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

        fig, axes = plt.subplots(1, 2, sharex=True)
        axes[0].plot(diffs[i]); axes[0].set_title('1st Differencing')
        axes[1].set(ylim=(0,5))
        plot_pacf(diffs[i], ax=axes[1])

        plt.show()

    # Now find the q parameters
    print("q Parameters")
    for i in range(K-2):
        # PACF plot of 1st differenced series
        plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

        fig, axes = plt.subplots(1, 2, sharex=True)
        axes[0].plot(diffs[i]); axes[0].set_title('1st Differencing')
        axes[1].set(ylim=(0,1.2))
        plot_acf(diffs[i], ax=axes[1])

        plt.show()
    '''

    p = np.array([1,1,1,7,1,7])
    q = np.array([1,2,1,7,1,7])

    stores = [0,1,2,3,4,6]

    
    for i in range(K-2):
        split_1 = int(np.round(0.75 * len(diffs[i])))
        split_2 = len(diffs[i]) - split_1
        train = diffs[i][:split_1]
        test = diffs[i][-split_2:]

        model = ARIMA(train, order=(p[i],d[i],q[i]))
        model_fit = model.fit(method_kwargs={'maxiter':300})
        
        # Forecast
        fc_len = 7

        fc = model_fit.get_forecast(fc_len)
        yhat = fc.predicted_mean
        yhat_conf_int = fc.conf_int(alpha=0.05)

        ind_1 = np.arange(split_1, split_1+fc_len, 1)
        ind_2 = np.arange(split_1, split_1+len(test))
        # Make as pandas series
        fc_series = pd.Series(yhat, index=ind_1)
        test = pd.Series(test, index=ind_2)
        lower_series = pd.Series(yhat_conf_int[:, 0], index=ind_1)
        upper_series = pd.Series(yhat_conf_int[:, 1], index=ind_1)

        name = 'ARIMAmodel_' + str(stores[i]) + '.pkl'

        model_fit.save(name)

    '''
    # Plot

    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(train, label='training')
    plt.plot(test, label='actual')
    plt.plot(fc_series, label='forecast')
    plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
    '''