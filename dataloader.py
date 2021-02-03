# coding: utf-8

# # Quarter 2: Project
# 
# The main deliverable for the second quarter of the computational economics and finance class is a project to showcase what you have learned. The goal of this project is for you (potentially in a small group) to produce a piece of work (in the form of a Jupyter notebook) that you would be able to use to showcase the skills that you have learned in this class to potential employers or academic advisors.
# 
# The relatively loose structure in self-directed projects like this makes them a bit more challenging than other things that you will do in school, but we think it also makes them more interesting. They give you a chance to indulge your curiosity and show off your creativity.
# 
# We have broken the project into three components to keep you on track. Each of these components should be turned in as its own Jupyter notebook. The first two steps are graded almost entirely on whether you do them or not. You must complete the first two steps on your own. When you actually begin working on the final project, you may work in groups of two to four, but you may also work alone if you’d  prefer.

# ## Project Ideas
# 
# **(15% of project grade)**
# 
# Form a team of between one and four students (no more) and choose a project. This project could come from the ideas that you submitted or some other idea if you get a sudden flash of inspiration. Flesh out the project in detail -- When we say “flesh it out in detail”, we mean properly load the data into the notebook, describe in words what you want to explore, and create a couple draft quality visualizations (don’t worry about making them pretty) that whets a reader’s appetite.
# 
# Please include the names of all group members in the below:
# 
# * Jian Zhou
# * Bo Sun
# * Man Chen
# 
# 
# **Please note that each person should make a copy of the notebook and turn it in!**
# 

# ## Proposal:
# 
# Include your project proposal, data, and graphs in the cells below

# ### Project Proposal

# The stochastic volatility model is the canonical method to model asset return volatility. Nonetheless, the stochastic models usually don’t have closed-form analytic solution and computational-heavy to calibrate due to model complexity. Monte Carlo Markov Chain has grown to be one of the most effective and popular tools in analyzing the stochastic volatility model. Priors and data are fed into the model, and posteriors are the calibrated model parameters of interest. 
# 
# We have noticed that the VIX index has been rather volatile since the outbreak of COVID-19 and peaked in March 2019. The traditional stochastic volatility fails to capture the spike due to the inactivity to internalize exogenous shock.  
# 
# We proposed to extend the canonical stochastic volatility to incorporate covid-19 data under a Monte Carlo Markov Chain framework. The following formulas set the extended stochastic volatility model.

# Mathematically speaking, the cannonical stochastic volatility model follows the below setting, where $\sigma_\eta$ is the scale parameter.

# $$\begin{align*}
# \qquad & y_t = \beta e^{h_t/2} \epsilon_t\qquad \qquad &\epsilon_t \sim N(0,1) \\
# \qquad & h_{t} = \mu + \phi (h_{t-1} - \mu) + \sigma_{\eta} \eta_{t-1}  \qquad \qquad &\eta_{t-1} \sim N(0,1) \quad \beta \sim N(1,10) \\
# \qquad &  h_1 \sim N(\mu, \sigma_\eta^2/(1-\phi^2)) \\
# \end{align*}$$<br>

# The extended stochastic volatility model follows the below setting. The key difference is that $\sigma_\eta$ is no more time-invariant, instead, is a random walk with drift decided by covid-19 data.

# $$\begin{align*}
# \qquad & y_t = \beta e^{h_t/2} \epsilon_t\qquad \qquad &\epsilon_t \sim N(0,1) \\
# \qquad & h_{t} = \mu + \phi (h_{t-1} - \mu) + \sigma_{\eta,t-1} \eta_{t-1}  \qquad \qquad &\eta_{t-1} \sim N(0,1) \quad \beta \sim N(1,10) \\
# \qquad &  h_1 \sim N(\mu, \sigma_\eta^2/(1-\phi^2)) \\
# \qquad & \color{red}{ \sigma_{\eta,t} =  \sigma_{\eta,t-1} + \alpha(lnC_t - lnC_{t-1}) + u_t} \qquad \qquad &u_t \sim N(0,1) \\
# \qquad &  \color{red} {\sigma_{\eta,1} \sim N(1, 10)}
# \end{align*}$$<br>

# We plan to sample and calibrate the extended stochastic volatility model. Furthermore, we will conduct a model comparison to check whether the comprehensive model supersedes the canonical model. 

# ### Prior

# In[1]:


import datetime
import os

import numpy as np
import pandas as pd
import pickle
import yfinance as yf

# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# $y_t$ is the mean corrected return
# $$\begin{align*}
# \qquad & y_t \sim N(0, e^{h_t})\\
# \qquad & h_{t} \sim N(\phi h_{t-1}, \sigma_{\eta}) \\
# \qquad &  h_1 \sim N(\mu, \sigma_\eta^2/(1-\phi^2)) \\
# \qquad & \phi \sim Beta(20,1.5) \\
# \qquad & \sigma^2_{\eta} \sim InverseGamma(2.5,0.05) \\
# \end{align*}$$<br>

# We set the beta(20, 1.5) so that the prior mean will be 20/21.5 = 0.86. The estimation is stationary process.
# 
# We set $\mu$ = 0. 
# 
# The mean of IG is alpha/beta, which is 50 in our cases
# We set $\mu$ = 0. 

# $$   f(x \mid \mu, \tau) =
#        \sqrt{\frac{\tau}{2\pi}}
#        \exp\left\{ -\frac{\tau}{2} (x-\mu)^2 \right\}$$
#        
# $$\tau = \dfrac{1}{\sigma^2}$$




# ### Data

def load_data(regular = True, reload=False):
    if not reload:
        if os.path.exists("../Data/data.pkl"):
            data = pd.read_pickle("../Data/data.pkl")
            return data
        else:
            return load_data(reload=True)

    # load sp500 and vix data use yfinance
    _finance_data = yf.download("^GSPC ^VIX", start="2017-01-01", end="2021-01-11")['Adj Close']
    _finance_data = _finance_data.rename({'^GSPC':'SP500', '^VIX':'VIX', '^INDIAVIX':'India_VIX'}, axis=1)
    
    
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
    
    
    # # calculate the cases growth in US
    # covid_19['cases_growth_US'] = np.log(covid_19['US']).diff().rolling(14, win_type='gaussian').mean(std=3)
    
    
    # calculate the cases growth globally
    covid_19_country['cases_growth_global'] = covid_19_country.mean(axis=1)
    
    covid_19_data = covid_19[['US', 'global']].merge(covid_19_country[['cases_growth_US', 
                                                                       'cases_growth_global', 
                                                                       'cases_growth_India']], 
                                                     left_index=True, right_index=True)
    
    
    covid_19_data.index = covid_19_data.reset_index()['index'].apply(lambda i : datetime.datetime.strptime(i, '%m/%d/%y'))
    

    # merge data 
    data = finance_data.merge(covid_19_data, left_index=True, right_index=True)
    if not os.path.exists("../Data"):
        os.mkdir("../Data")
    data.to_pickle("../Data/data.pkl")
    return data


if __name__ == '__main__':
    data = load_data()
    print (data.tail(3))
