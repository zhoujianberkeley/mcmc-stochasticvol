import numpy as np
import pymc3 as pm

from dataloader import load_data

data = load_data()

def make_baseline_model_AR1(data, observe):
    with pm.Model() as model:
        #prior
        scale = pm.InverseGamma("scale", alpha=2.5, beta=0.05)
        phi = pm.Beta("phi", alpha=20, beta=1.5)
        # log_vol = pm.GaussianRandomWalk('log_vol', mu=0, sigma=np.sqrt(simga_ita2), shape=len(data))
        log_vol = pm.AR1("log_vol", k=phi, tau_e=1/scale, shape=len(data))
        returns = pm.Normal("returns", mu=0, sigma=np.exp(log_vol/2), observed=observe)
    return model


def make_baseline_model_RW(data, observe):
    with pm.Model() as model:
        step_size = pm.Exponential("step_size", 10)
        volatility = pm.GaussianRandomWalk("volatility", sigma=step_size, shape=len(data))
        nu = pm.Exponential("nu", 0.1)
        returns = pm.StudentT(
            "returns", nu=nu, lam=np.exp(-2 * volatility), observed=observe)
    return model

def make_covid_model(data, observe):
    data = data.dropna()
    if type(observe) is str:
        observe = data[observe]
    with pm.Model() as model:
        # data
        _covid = pm.Data("covid", data.cases_growth_US)
        # prior

        phi = pm.Beta("phi", alpha=20, beta=1.5)
        alpha = pm.Flat('alpha')
        scale = pm.GaussianRandomWalk("scale", mu=alpha*_covid, sigma=1, shape=len(data))
        log_vol = pm.AR1("log_vol", k=phi, tau_e=1/scale**2, shape=len(data))

        # likilihood
        returns = pm.Normal("returns", mu=0, sigma=np.exp(log_vol/2), observed=observe)
    return model

if __name__ == '__main__':
    # freeze_support()
    # returns = pd.read_csv(pm.get_data("SP500.csv"), index_col="Date")
    # returns["change"] = np.log(returns["Close"]).diff()
    # returns = returns.dropna()
    # returns.head()

    baseline_model = make_covid_model(data, "log_returns")

    with baseline_model:
        prior = pm.sample_prior_predictive(500)

    with baseline_model:
        trace = pm.sample(2000, tune=2000)
