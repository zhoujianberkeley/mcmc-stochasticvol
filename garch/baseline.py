# import numpy as np
# import pymc3 as pm

# from dataloader import load_data

# data = load_data()

# def garch_baseline_model(data):
#     with pm.Model() as model:
#         omega = pm.InverseGamma("omega", alpha=2.5, beta=0.05)
#         alpha1 = pm.Gamma("alpha1", alpha=1, beta=1)
#         beta1 = pm.Gamma("beta1", alpha=1, beta=1)
#         log_vol = pm.Garch11('log_vol', omega=omega, alpha1=alpha1/2, beta1=beta1/2, shape=len(data))
#         returns = pm.Normal("returns", mu=0, sigma=np.exp(log_vol/2), observed=data["SP500"])
#     return model

# baseline_model = garch_baseline_model(data)

# with baseline_model:
#     prior = pm.sample_prior_predictive(500)

# with baseline_model:
#     trace = pm.sample(20, tune=20)


import numpy as np
import arch as ar
from dataloader import load_data

data = load_data()
data['returns'] = 100*(data.SP500 / data.SP500.shift(1) - 1).fillna(0)
returns = data.returns



from matplotlib import pyplot
from arch import arch_model
# seed pseudorandom number generator
# split into train/test
n_test = 40
train, test = data.returns.iloc[:-n_test],data.returns.iloc[-n_test:]
# define model
model = arch_model(train, mean='Zero', vol='GARCH', p=3, q=3)
# fit model
model_fit = model.fit()
# forecast the test set
yhat = model_fit.forecast(horizon=n_test)
# plot the actual variance
# plot forecast variance
pyplot.plot(yhat.variance.values[-1, :])
pyplot.show()