import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm

from dataloader import load_data

data = load_data(reload=False)

def make_baseline_model_RW(data, observe):
    with pm.Model() as model:
        # Piror
        nu = pm.Exponential("nu", 0.1)
        scale = pm.Exponential("scale", 10)
        volatility = pm.GaussianRandomWalk("volatility", sigma=scale, shape=len(data))
        # Likelihood
        returns = pm.StudentT("returns", nu=nu, lam=np.exp(-2 * volatility), observed=data[observe])
    return model

def make_baseline_model_AR1(data, observe):
    with pm.Model() as model:
        # Piror
        phi = pm.Beta("phi", alpha=20, beta=1.5)
        scale = pm.InverseGamma("scale", alpha=2.5, beta=0.05)
        _log_vol = pm.AR1("_log_vol", k=phi, tau_e=1/scale, shape=len(data))
        mu = pm.Exponential('mu', lam=0.1)
        log_vol = pm.Deterministic("log_vol", _log_vol + mu)
        # Likelihood
        returns = pm.Normal("returns", mu=0, sigma=np.exp(log_vol/2), observed=data[observe])
    return model

def make_state_model_AR1(data, observe):
    # Prepare data
    nstate = data['covid_state_US'].nunique()
    log_returns = data[observe].to_numpy()
    state_idx = data["covid_state_US"].to_numpy()

    with pm.Model() as model:
        # Data
        _returns = pm.Data("_returns", log_returns)
        _state_idx = pm.intX(pm.Data("state_idx", state_idx))
        # Prior
        scale = pm.InverseGamma("scale", alpha=2.5, beta=0.05, shape=nstate)
        phi = pm.Beta("phi", alpha=20, beta=1.5)
        # log_vol = pm.GaussianRandomWalk('log_vol', mu=0, sigma=np.sqrt(scale[_state_idx]), shape=len(data))
        log_vol = pm.AR1("log_vol", k=phi, tau_e=1/scale[_state_idx], shape=len(data)+1)
        # Likelihood
        returns = pm.Normal("returns", mu=0, sigma=np.exp(log_vol[1:]/2), observed=_returns)
    return model


def make_covid_model(data, observe_str):
    data = data.dropna()

    if type(observe_str) is str:
        _observe = data[observe_str]
    with pm.Model() as model:
        # Data
        _change_returns = pm.Data("_change_returns", data[observe_str], dims=observe_str, export_index_as_coords=True)
        _covid = pm.Data("covid", data.cases_growth_US)
        # HyperPrior
        alpha = pm.Normal("alpha", mu=0, sigma=1, testval=np.random.randn())
        scale = pm.GaussianRandomWalk("scale", mu=alpha*_covid, sigma=1, shape=len(data), testval=np.random.randn(len(data)))
        # Prior
        phi = pm.Beta("phi", alpha=20, beta=1.5)
        log_vol = pm.AR1("log_vol", k=phi, tau_e=1 / (scale ** 2), shape=len(data)+1, testval=np.random.randn(len(data)+1))
        # Likilihood
        returns = pm.Normal("returns", mu=0, sigma=pm.math.exp(log_vol[1:]/2), dims=observe_str, observed=_observe)
    return model


if __name__ == '__main__':
    # freeze_support()

    baseline_model = make_baseline_model_AR1(data, "log_returns")

    with baseline_model:
        # step = pm.Metropolis()
        # trace = pm.sample(4000, tune=3000, step=step, return_inferencedata=True)
        trace = pm.sample(4000, tune=3000, return_inferencedata=True)

    with baseline_model:
        az.plot_trace(
            trace, var_names=["phi", "scale", "mu"],
            compact=True)
    plt.show()