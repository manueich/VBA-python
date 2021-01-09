import numpy as np


def f_model(X, th, u, inF):
    # Function defining the ODE evolution of the van der Pol oscillator

    n = 2               # Model Order
    n_theta = 1         # No of parameters

    # Model Equations
    dx = np.zeros((n, 1))
    dx[0, 0] = X[1]
    dx[1, 0] = th[0]*(1 - X[0]**2)*X[1] - X[0]

    # Derivatives of the ODEs w.r.t
        # Model states
    dFdX = np.zeros((n, n))
    dFdX[0, 0] = 0
    dFdX[0, 1] = 1
    dFdX[1, 0] = -2*th[0]*X[0]*X[1]-1
    dFdX[1, 1] = th[0]*(1 - X[0]**2)
        # Evolution parameters
    dFdTh = np.zeros((n, n_theta))
    dFdTh[0, 0] = 0
    dFdTh[1, 0] = (1 - X[0]**2)*X[1]

    return dx, dFdX, dFdTh


def f_obs(X, phi, u, inG):
    # Observation function defining a logistic mapping with unknown slope

    n_phi = 1       # No of Observation Parameters
    nY = 1          # No of Observations
    n = 2           # Model order

    # Observation Equation
    gx = np.zeros((nY, 1))
    gx[0, 0] = 1 / (1 + np.exp(-phi[0] * X[0]))

    # Derivatives of the Observation equation w.r.t
        # - Model states
    dGdX = np.zeros((nY, n))
    dGdX[0, 0] = phi[0] * np.exp(-phi[0] * X[0]) / ((np.exp(-phi[0] * X[0]) + 1)**2)
    dGdX[0, 1] = 0
        # - Observation Parameters
    dGdPhi = np.zeros((nY, n_phi))
    dGdPhi[0, 0] = X[0] * np.exp(-phi[0] * X[0]) / ((np.exp(-phi[0] * X[0]) + 1)**2)

    return gx, dGdX, dGdPhi
