import numpy as np
from Functions import VBA
from Demos.LinearDampedOscillator import f_LinDamOsc as f_model

# -------------- SIMULATE MODEL ------------------------------

# Integration Time grid
dt = 0.5
t = np.arange(0, 250 + dt, dt)

# Time points at which to provide data
ty = np.arange(1, 250+1, 10)
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
           "ODESolver": 'RK',
           "inF": inF,
           "dim": dim,
           "Display": True}

# Simulate Data
yd = VBA.simulate(ty, t, [], priors, options, True)

# ---------------------------------------------

# Change Prior structure
priors.update({"muTheta": 0.1*np.ones((dim["n_theta"], 1))})
priors.update({"muX0": 0.1*np.ones((dim["n"], 1))})
priors.update({"a": 1,
               "b": 1})

data = {"y": yd,
        "t": ty}

# Call Inversion routine
posterior, out = VBA.main(data, t, priors, options)

