#!/usr/bin/env python
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import yfinance as yf
from IPython import get_ipython

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

# We load financial data from yahoo finance

# In[4]:


# load sp500 and vix data use yfinance
_finance_data = yf.download("^GSPC ^VIX", start="2017-01-01", end="2021-01-11")['Adj Close']
_finance_data = _finance_data.rename({'^GSPC':'SP500', '^VIX':'VIX', '^INDIAVIX':'India_VIX'}, axis=1)
_finance_data.head(2)


# In[5]:


VIX_india = pd.read_csv("indian vix.csv").pipe(pd.DataFrame.rename, columns=lambda x: x.strip()) .pipe(pd.DataFrame.rename, {'Close':'India_VIX'}, axis=1) .pipe(pd.DataFrame.set_index, ['Date']) 
VIX_india.index = VIX_india.reset_index()['Date'].apply(lambda i : datetime.datetime.strptime(i, '%d-%b-%y'))
finance_data = _finance_data.merge(VIX_india, left_index=True, right_index=True)
finance_data.head(2)


# We load covid-19 data from Johns Hopkins Coronavirus Resource Center.
# The raw data is taken a gaussian smoothing with a 2-week window size and standard deviation 3.

# In[6]:


# load covid-19 data from Johns Hopkins Coronavirus Resource Center
_covid_19 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",
           error_bad_lines=False)


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
covid_19_data.tail(2)


# In[7]:


# merge data 
data = finance_data.merge(covid_19_data, left_index=True, right_index=True)
data.tail(3)


# ### Graphs

# #### VIX peaks at March, 2020, as the covid-19 panics the street. 
# 
# Furthermore, we notice that VIX sicne covid-19 has been stably higher than before, implying some systematic change.
# 

# In[8]:


finance_data.plot(y=['VIX', 'India_VIX'], use_index=True, figsize=(15,5))


# #### The log-difference of the confirmed cases of covid-19 (2-week gaussian smoothed) spiked in March, 2020 in US and around the world.

# In[9]:


fig,ax = plt.subplots(figsize=(15,5))
# make a plot
ax.plot(data.index, data.loc[:, 'cases_growth_US'], label="US")
ax.plot(data.index, data.loc[:, 'cases_growth_global'], label="Global")
ax.plot(data.index, data.loc[:, 'cases_growth_India'], label="India")
ax.set_ylabel("log-difference of the confirmed cases", fontsize=14)
plt.legend()


# #### There is a salient overlap in March, 2020, if we plot VIX and the log-difference of the confirmed cases together.
# 
# The common co-movements inspire our project idea - incorporating covid-19 data with the stochastic model. We believe the overlap of peaks is not by coincidence, while indeed, covid-19 cases can provide some insightful information about the scale parameter $\sigma_\eta$

# In[10]:


fig,ax = plt.subplots(figsize=(15,5))
# make a plot
ax.plot(data.index, data.VIX, color="red")
# set x-axis label
ax.set_xlabel("Time", fontsize=14)
# set y-axis label
ax.set_ylabel("VIX", fontsize=14)

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(data.index, data['cases_growth_US'], label="US")
ax2.plot(data.index, data['cases_growth_global'], label='Global')
ax2.set_ylabel("log-difference of the confirmed cases", fontsize=14)
plt.legend()
plt.show()


# In[11]:


fig,ax = plt.subplots(figsize=(15,5))
# make a plot
ax.plot(data.index, data.India_VIX, color="red")
# set x-axis label
ax.set_xlabel("Time", fontsize=14)
# set y-axis label
ax.set_ylabel("India_VIX", fontsize=14)

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(data.index, data['cases_growth_India'], label="India")
ax2.plot(data.index, data['cases_growth_global'], label='Global')
#ax2.plot(data.index, data['cases_growth_India'], label='India')
ax2.set_ylabel("log-difference of the confirmed cases", fontsize=14)
plt.legend()
plt.show()


# ### Baseline Model: Stochastic Volatility

# In[15]:


data


# In[24]:

def make_baseline_model(data):
    with pm.Model() as model:
        simga_ita2 = pm.InverseGamma("simga_ita2", alpha=2.5, beta=0.05)
        phi = pm.Beta("phi", alpha=20, beta=1.5)
        log_vol = pm.AR1("log_vol", k=phi, tau_e=1/simga_ita2, shape=len(data))
        returns = pm.Normal("returns", mu=0, sigma=np.exp(log_vol/2), observed=data["SP500"])
    return model

baseline_model = make_baseline_model(data)
# init=pm.Normal(name='start', mu=0, sigma=(0.05/1.5)/(1-(20/21.5)**2)),
# In[ ]:

pm.model_to_graphviz(baseline_model)

# In[ ]:
with baseline_model:
    prior = pm.sample_prior_predictive(500)

# In[ ]:

with baseline_model:
    trace = pm.sample(2000, tune=2000)
