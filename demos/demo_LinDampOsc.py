import numpy as np
from Functions import VBA
from Functions import VBA_simulate
from Models import f_LinDamOsc as f_model
import matplotlib.pyplot as plt

# -------------- SIMULATE MODEL ------------------------------

# Integration Time grid
dt = 0.5
t = np.arange(0, 200 + dt, dt)

# Time points at which to provide data
ty = np.arange(1, 200+1, 10)
ty = ty.reshape(1, ty.size)

# Model Dimensions
dim = {"n": 2,          # Model order
       "n_theta": 2,    # No of evolution parameters
       "n_phi": 0}      # No of observation parameters

# Integration time step
inF = {"dt": dt}

# Create prior structure
    # - Evolution Parameters
muTheta = np.zeros((dim["n_theta"], 1))
muTheta[0] = 0.02
muTheta[1] = 0.5
SigmaTheta = np.eye(muTheta.size)

    # - Initial Conditions
muX0 = np.zeros((dim["n"], 1))
muX0[0] = 0
muX0[1] = 0.1
SigmaX0 = np.eye(muX0.size)

    # - Noise parameters
a = 50
b = 1

priors = {"a": a,
          "b": b,
          "muTheta": muTheta,
          "SigmaTheta": SigmaTheta,
          "muX0": muX0,
          "SigmaX0": SigmaX0}

# Define Options
options = {"f_model": f_model.f_model,
           "f_obs": f_model.f_obs,
           "ODESolver": 'Euler',
           "inF": inF,
           "dim": dim}

# Simulate Data
yd = VBA_simulate.simulate(ty, t, [], priors, options)

# Change Prior structure
priors.update({"muTheta": 0.5*np.ones((dim["n_theta"], 1))})

data = {"y": yd,
        "t": ty}

posterior, out = VBA.main(data, t, priors, options)

