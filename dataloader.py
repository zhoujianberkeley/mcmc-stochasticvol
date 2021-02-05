import datetime
import os

import numpy as np
import pandas as pd
import pickle
import yfinance as yf

def load_data(regular = True, reload=False):
    '''
    prepare data for the project
    :param regular: for developers only
    :param reload: reload data online
    :return: a dataframe
    '''
    if not reload:
        if os.path.exists("../Data/data.pkl"):
            data = pd.read_pickle("../Data/data.pkl")
            return data
        else:
            return load_data(reload=True)

    # load sp500 and vix data use yfinance
    _finance_data = yf.download("^GSPC ^VIX", start="2017-01-01", end="2021-01-11")['Adj Close']
    _finance_data = _finance_data.rename({'^GSPC':'SP500', '^VIX':'VIX', '^INDIAVIX':'India_VIX'}, axis=1)
    _finance_data['log_returns'] = np.log(_finance_data['SP500']).diff()
    _finance_data['returns'] = _finance_data['SP500'].pct_change()
    _finance_data['real_3w_vol'] = _finance_data['returns'].rolling(window=15).apply(pd.DataFrame.std)

    try:
        VIX_india = pd.read_csv("indian vix.csv").pipe(pd.DataFrame.rename, columns=lambda x: x.strip()) .pipe(
            pd.DataFrame.rename, {'Close':'India_VIX'}, axis=1) .pipe(pd.DataFrame.set_index, ['Date'])
    except FileNotFoundError:
        VIX_india = pd.read_csv("../indian vix.csv").pipe(pd.DataFrame.rename, columns=lambda x: x.strip()).pipe(
            pd.DataFrame.rename, {'Close': 'India_VIX'}, axis=1).pipe(pd.DataFrame.set_index, ['Date'])
    VIX_india.index = VIX_india.reset_index()['Date'].apply(lambda i : datetime.datetime.strptime(i, '%d-%b-%y'))
    finance_data = _finance_data.merge(VIX_india, left_index=True, right_index=True)



    # We load covid-19 data from Johns Hopkins Coronavirus Resource Center.
    # The raw data is taken a gaussian smoothing with a 2-week window size and standard deviation 3.

    # load covid-19 data from Johns Hopkins Coronavirus Resource Center
    # code for Bo Sun only
    if regular:
        _covid_19 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",
                error_bad_lines=False)
    else:
        _covid_19 = pd.read_csv(r"C:\Users\harvey_sun\Desktop\data\time_series_covid19_confirmed_global.csv")


    covid_19 = _covid_19.pipe(pd.DataFrame.drop, ['Province/State', 'Lat', 'Long'], 1)         .pipe(pd.DataFrame.set_index, 'Country/Region').T

    covid_19['global'] = covid_19.apply('sum', axis=1)

    # calculate the cases growth in each country
    covid_19_country = pd.DataFrame(index=covid_19.index)
    for country in covid_19.columns.unique():
            covid_19_country['cases_growth_' + country] = np.log(covid_19[[country]].sum(axis=1).replace(0,1)).diff().rolling(14, win_type='gaussian').mean(std=3)

    # calculate the cases growth globally
    covid_19_country['cases_growth_global'] = covid_19_country.mean(axis=1)

    covid_19_data = covid_19[['US', 'global']].merge(covid_19_country[['cases_growth_US',
                                                                       'cases_growth_global',
                                                                       'cases_growth_India']],
                                                     left_index=True, right_index=True)
    # set covid state for US, use 0.05 as threshold line
    covid_19_data.loc[:, 'covid_state_US'] = 0
    covid_19_data.loc[covid_19_data.cases_growth_US > 0.05, 'covid_state_US'] = 1

    covid_19_data.index = covid_19_data.reset_index()['index'].apply(lambda i : datetime.datetime.strptime(i, '%m/%d/%y'))

    # merge data
    data = finance_data.merge(covid_19_data, left_index=True, right_index=True)
    if not os.path.exists("../Data"):
        os.mkdir("../Data")
    data.to_pickle("../Data/data.pkl")
    return data


if __name__ == '__main__':
    data = load_data(reload=True)
    print (data.tail(3))
