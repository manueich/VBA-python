import numpy as np
from Functions import VBA
from Demos.VanDerPolOscillator import f_VanDerPol as f_model

# Demo that illustrates the inversion of a model describing a van der Pol oscillator
# partially observed through a logistic mapping

# -------------- SIMULATE MODEL ------------------------------

# Integration Time grid
dt = 0.05
t = np.arange(0, 12 + dt, dt)
t = t.reshape(1, t.size)

# Time points at which to provide data
ty = np.arange(1, 12+0.2, 0.2)
ty = ty.reshape(1, ty.size)

# Model Dimensions
dim = {"n": 2,          # Model order
       "n_theta": 1,    # No of evolution parameters
       "n_phi": 1}      # No of observation parameters

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

    # - Observation Parameters
muPhi = np.zeros((dim["n_phi"], 1))
muPhi[0] = 2
SigmaPhi = np.eye(muTheta.size)

    # - Noise parameters
a = 500
b = 1

# Priors for simulation
priors_sim = {"a": a,
              "b": b,
              "muTheta": muTheta,
              "SigmaTheta": SigmaTheta,
              "muX0": muX0,
              "SigmaX0": SigmaX0,
              "muPhi": muPhi,
              "SigmaPhi": SigmaPhi}

# Define Options
options = {"f_model": f_model.f_model,
           "f_obs": f_model.f_obs,
           "dim": dim,
           "Display": True}

# Simulate Data
yd = VBA.simulate(ty, t, [], priors_sim, options, True)

# -------- MODEL INVERSION --------------
# Set data
data = {"y": yd,
        "t": ty}

# Change Prior structure for model inversion
priors = priors_sim.copy()
priors.update({"muTheta": 0.1*np.ones((dim["n_theta"], 1))})
priors.update({"muPhi": 0.1*np.ones((dim["n_theta"], 1))})
priors.update({"muX0": 0.1*np.ones((dim["n"], 1))})
priors.update({"a": 0.1})
priors.update({"b": 0.1})

# Call Inversion routine
posterior, out = VBA.main(data, t, priors, options)

# Compare inferred posterior to true values used for the simulation
VBA.compare_to_sim(posterior, priors_sim, options)
