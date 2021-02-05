import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm

from dataloader import load_data

data = load_data(reload=False)

def make_baseline_model_RW(data, observe):
    '''
    model for Random Walk StoVol

    :param data: observation data
    :param observe: column name of y
    :return: PyMC model
    '''
    with pm.Model() as model:
        # Piror
        nu = pm.Exponential("nu", 0.1)
        scale = pm.Exponential("scale", 10)
        log_vol = pm.GaussianRandomWalk("log_vol", sigma=scale, shape=len(data))
        # Likelihood
        returns = pm.StudentT("returns", nu=nu, lam=np.exp(-2 * log_vol), observed=data[observe])
    return model

def make_baseline_model_AR1(data, observe):
    '''
    model for AR1 StoVol

    :param data: observation data
    :param observe: column name of y
    :return: PyMC model
    '''
    with pm.Model() as model:
        # Piror
        # phi = pm.Beta("phi", alpha=20, beta=1.5)
        np.random.seed(12345)
        phi = pm.Normal("phi", mu=1, sigma=1, testval=np.random.randn())
        # scale = pm.InverseGamma("scale", alpha=2.5, beta=0.05)
        scale = pm.Exponential("scale", 10, testval=np.random.randn())
        _log_vol = pm.AR1("_log_vol", k=phi, tau_e=1/pm.math.sqr(scale), shape=len(data), testval=np.random.randn(len(data)))
        # mu = pm.Exponential('mu', lam=0.1)
        mu = pm.Normal('mu', mu=0, sigma=1)
        log_vol = pm.Deterministic("log_vol", _log_vol + mu)
        mean_return = pm.Normal("mean_return", 0, 1)
        nu = pm.Exponential("nu", 0.1)
        # Likelihood
        # returns = pm.Normal("returns", mu=mean_return, sigma=np.exp(log_vol/2), observed=data[observe])
        returns = pm.StudentT("returns", nu=nu, mu=mean_return, lam=np.exp(-2 * log_vol), observed=data[observe])
    return model

def make_state_model_AR1(data, observe):
    '''
    model for Two-State StoVol

    :param data: observation data
    :param observe: column name of y
    :return: PyMC model
    '''
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
        log_vol = pm.GaussianRandomWalk('log_vol', mu=0, sigma=scale[_state_idx], shape=len(data))
        nu = pm.Exponential("nu", 0.1)
        # Likelihood
        returns = pm.StudentT("returns", nu=nu, lam=np.exp(-2 * log_vol), observed=_returns)
    return model


def make_covid_model(data, observe, col_covid="cases_growth_US", process='GRW'):
    '''
    model for Free-Scale StoVol

    :param data: observation data
    :param observe: column name of y
    :param col_covid: column name of covid data
    :param process: process of the scale paramter, can be GRW or AR1
    :return: PyMC model
    '''

    if data[col_covid].hasnans:
        raise ValueError(f"{col_covid} has NaN values")

    log_returns = data[observe].to_numpy()

    with pm.Model() as model:
        # Data
        _returns = pm.Data("_returns", log_returns)
        # _change_returns = pm.Data("_change_returns", data[observe_str], dims=observe_str, export_index_as_coords=True)
        _covid = pm.Data("covid", data[col_covid])

        # HyperPrior
        alpha = pm.Normal("alpha", mu=1, sigma=1, testval=np.random.random())
        scale = pm.GaussianRandomWalk("scale", mu=alpha*_covid, sigma=1, shape=len(data),
                                      testval=np.random.randint(low=1, high=10, size=len(data)))
        # Prior
        if process == 'GRW':  # scale follows a Gaussian Random Walk
            log_vol = pm.GaussianRandomWalk("log_vol", sigma=scale, shape=len(data),
                                        testval=np.random.randint(low=1, high=10, size=len(data)))
        elif process == 'AR1': # scale follows a AR1
            phi = pm.Beta("phi", alpha=20, beta=1.5)
            # phi = pm.Normal("phi", mu=1, sigma=1, testval=np.random.randint(low=1, high=10))
            log_vol = pm.AR1("log_vol", k=phi, tau_e=1 / pm.math.sqr(scale), shape=len(data)+1,
                             testval=np.random.randint(low=1, high=10, size=len(data)+1))[:-1]
        nu = pm.Exponential("nu", 0.1)

        # Likilihood
        returns = pm.StudentT("returns", nu=nu, lam=np.exp(-2 * log_vol), observed=_returns)
    return model


def model_diagnose(model, trace, var_names):
    '''
    diagnose a model based on 'Effective Sample Size and Rhat

    :param model: a PyMC3 model
    :param trace: sample trace
    :param var_names: variable names
    :return: None
    '''
    ess = az.ess(trace, relative=True)

    print("Effective Sample Size (min across parameters)")
    for var in var_names:
        print(f"\t{var}: {ess[var].values.min()}")
    rhat = az.rhat(trace)

    print("rhat (max across parameters)")
    for var in var_names:
        print(f"\t{var}: {rhat[var].values.max()}")

def gen_xy(trace, _data, y="log_vol", AR=False, skip=5):
    '''
    generate posterior predictive y from trace

    :param trace: sample trace
    :param _data: observation data
    :param y: y column anme
    :param AR: whether it's a AR model
    :param skip: how often take draws from trace, only take 1 draw after skip steps
    :return: x values from observation data, y values from posterior predictive
    '''
    _y_vals = np.exp(trace.posterior[y])
    y_vals = np.vstack([_y_vals[i] for i in range(_y_vals.shape[0])]).T
    if AR:  # take the last n-1 values, because AR1 process has an extra starting point
        y_vals = y_vals[:-1, ]
    # only take 1 draw after skip steps
    y_vals = y_vals.T[::skip].T
    x_vals = np.vstack([_data.index for _ in y_vals.T]).T.astype(np.datetime64)
    return x_vals, y_vals

def model_plot(data, trace, pp, AR=False):
    '''
    plot returns and volatility
    :param data: observation data
    :param trace: sample trace
    :param pp: posterior predictive
    :param AR: whether it's a AR model
    :return: figures
    '''
    x = data.index.to_numpy().astype(np.datetime64)
    fig, ax = plt.subplots(2, 1, figsize=(14, 8))

    # Plot returns
    ax[0].plot(
        x, pp["returns"][::10].T, color="g",
        alpha=0.25, zorder=-10
    )
    ax[0].plot(x, data["log_returns"].to_numpy(), color="k", linewidth=2.5, label="log returns")
    ax[0].set(title="Posterior predictive log-returns (green) VS actual log-returns", ylabel="Log Returns")

    # Plot volatility
    _, y_vals = gen_xy(trace, data, AR=AR, skip=10)
    ax[1].plot(x, y_vals, "k", alpha=0.01)
    ax[1].plot(data.index, data['real_3w_vol'], linewidth=2.5, label="realized 3W vol")
    ax[1].set(title="Estimated volatility over time (balck) vs realized vol", ylabel="Volatility")
    ax[1].set_ylim(bottom=0)

    # Add legends
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    plt.tight_layout()

if __name__ == '__main__':
    # freeze_support()
    baseline_model = make_state_model_AR1(data, "log_returns")
    with baseline_model:
        # step = pm.Metropolis()
        # trace = pm.sample(4000, tune=3000, step=step, return_inferencedata=True)
        trace = pm.sample(4000, tune=3000, return_inferencedata=True)

    with baseline_model:
        az.plot_trace(
            trace, var_names=["phi", "scale", "mu"],
            compact=True)
    plt.show()