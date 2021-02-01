import numpy as np
import pymc3 as pm

from dataloader import load_data

data = load_data()

def make_baseline_model(data):
    with pm.Model() as model:
        simga_ita2 = pm.InverseGamma("simga_ita2", alpha=2.5, beta=0.05)
        phi = pm.Beta("phi", alpha=20, beta=1.5)
        log_vol = pm.GaussianRandomWalk('log_vol', mu=0, sigma=np.sqrt(simga_ita2), shape=len(data))
        # log_vol = pm.AR1("log_vol", k=phi, tau_e=1/simga_ita2, shape=len(data))
        returns = pm.Normal("returns", mu=0, sigma=np.exp(log_vol/2), observed=data["SP500"])
    return model

baseline_model = make_baseline_model(data)

with baseline_model:
    prior = pm.sample_prior_predictive(500)

with baseline_model:
    trace = pm.sample(2000, tune=2000)
