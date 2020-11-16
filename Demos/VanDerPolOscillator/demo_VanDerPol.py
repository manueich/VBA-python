import numpy as np
from Functions import VBA
from Demos.VanDerPolOscillator import f_VanDerPol as f_model

# -------------- SIMULATE MODEL ------------------------------

# Integration Time grid
dt = 0.1
t = np.arange(0, 10 + dt, dt)
t = t.reshape(1, t.size)

# Time points at which to provide data
ty = np.arange(1, 10+0.2, 0.2)
ty = ty.reshape(1, ty.size)

# Input
u = np.zeros((2, np.shape(t)[1]))

# Model Dimensions
dim = {"n": 2,          # Model order
       "n_theta": 1,    # No of evolution parameters
       "n_phi": 0}      # No of observation parameters

# Integration time step
inG = {"scale": 1,
       "slope": 2}

# Create prior structure
    # - Evolution Parameters
muTheta = np.zeros((dim["n_theta"], 1))
muTheta[0] = 1
SigmaTheta = np.eye(muTheta.size)

    # - Initial Conditions
muX0 = np.zeros((dim["n"], 1))
muX0[0] = 1
muX0[1] = 1
SigmaX0 = np.eye(muX0.size)

    # - Noise parameters
a = 1000
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
           "inG": inG,
           "dim": dim,
           "Display": True}

# Simulate Data
yd = VBA.simulate(ty, t, u, priors, options, True)

# ---------------------------------------------

# Change Prior structure
priors.update({"muTheta": 0.1*np.ones((dim["n_theta"], 1))})
priors.update({"muX0": 0.1*np.ones((dim["n"], 1))})
priors.update({"a": 1})
priors.update({"b": 1})

data = {"y": yd,
        "t": ty,
        "u": u}

# Call Inversion routine
posterior, out = VBA.main(data, t, priors, options)
print("dONE")