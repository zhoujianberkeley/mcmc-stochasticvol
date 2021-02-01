import numpy as np
import pymc3 as pm

from dataloader import load_data

data = load_data()

def garch_baseline_model(data):
    with pm.Model() as model:
        omega = pm.InverseGamma("omega", alpha=2.5, beta=0.05)
        alpha1 = pm.Gamma("alpha1", alpha=1, beta=1)
        beta1 = pm.Gamma("beta1", alpha=1, beta=1)
        log_vol = pm.Garch11('log_vol', omega=omega, alpha1=alpha1/2, beta1=beta1/2, shape=len(data))
        returns = pm.Normal("returns", mu=0, sigma=np.exp(log_vol/2), observed=data["SP500"])
    return model

baseline_model = garch_baseline_model(data)

with baseline_model:
    prior = pm.sample_prior_predictive(500)

with baseline_model:
    trace = pm.sample(20, tune=20)
