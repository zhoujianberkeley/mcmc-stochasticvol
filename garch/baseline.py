 import numpy as np
 import pymc3 as pm

 from dataloader import load_data

 data = load_data()
 data['returns'] = 100*(data.SP500 / data.SP500.shift(1) - 1).fillna(0)

 def garch_baseline_model(data):
     with pm.Model() as model:
         omega = pm.InverseGamma("omega", alpha=2.5, beta=0.05)
         alpha1 = pm.Uniform("alpha1", 0, 1)
         beta1 = pm.Uniform("beta1", 0, 1)
         vol = pm.InverseGamma("omega", alpha=2.5, beta=0.05)
         returns = pm.GARCH11('returns', omega=omega, alpha1=alpha1, beta1=beta1, initial_vol=vol, shape=len(data), observed=data['returns'])
     return model

 baseline_model = garch_baseline_model(data)

 with baseline_model:
     prior = pm.sample_prior_predictive(500)

 with baseline_model:
     trace = pm.sample(200, tune=200)
