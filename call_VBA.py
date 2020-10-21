import numpy as np
from Functions import VBA
import f_model_2 as f_model
import matplotlib.pyplot as plt

# Create Data
ty = np.arange(1, 200+1, 1)
ty = ty.reshape(1, ty.size)
y = np.zeros((1, ty.size))
y[0, :] = 0.1*(ty**2)*np.exp(-0.01*(ty))
# y[1, :] = np.concatenate((np.array([0]), np.diff(y[0, :])),0)

# Integration Time grid
dt = 1
t = np.arange(0, 200 + dt, dt)

# Input
u = np.zeros((2, t.size))

data = {"y": y,
        "t": ty,
        "u": u}

# Create priors
    # - Observation Parameters
muPhi = np.zeros((1, 1))
muPhi[0] = 10
SigmaPhi = np.eye(muPhi.size)*1

    # - Evolution Parameters
muTheta = np.zeros((3, 1))
muTheta[0] = 0.02
muTheta[1] = 0.5
muTheta[2] = 0
SigmaTheta = np.eye(muTheta.size)*2

    # - Initial Conditions
muX0 = np.zeros((2, 1))
muX0[0] = 0
muX0[1] = 0.1
SigmaX0 = np.eye(muX0.size)*3

    # - Noise parameters
a = 2
b = 0.1

iQy = [np.array([[1, 0], [0, 2]])]
for i in range(0, 200 - 1):
    iQy.append(np.array([[1, 0], [0, 2]]))

priors = {"a": a,
          "b": b,
          "muPhi": muPhi,
          "SigmaPhi": SigmaPhi,
          "muTheta": muTheta,
          "SigmaTheta": SigmaTheta,
          "muX0": muX0,
          "SigmaX0": SigmaX0}

# Options
inF = {"dt": dt}
inG = []

dim = {"n": 2,
       "n_theta": 3,
       "n_phi": 0}

options = {"GnMaxIter": 100,
           "GnTolFun": 0.01,
           "MaxIter": 100,
           "TolFun": 0.01,
           "f_model": f_model.f_model,
           "f_obs": f_model.f_obs,
           "inF": inF,
           "inG": inG,
           "dim": dim,
           "updateHP": True,
           "verbose": True}


posterior, out = VBA.main(data, t, priors, options)

yout = posterior["ModelOut"]["y"]

fig, ax = plt.subplots()
plt.plot(ty[0, :],y[0, :], 'k', marker='o', ls='none')
plt.plot(t,yout[0, :],'r')
plt.show()

