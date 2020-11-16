import numpy as np

def f_model(X, th, u, inF):
    # Function defining the ODE evolution of the van der Pol oscillator

    n = 2               # Model Order
    n_theta = 1         # No of parameters

    # Model Equations
    dx = np.zeros((n, 1))
    dx[0, 0] = X[1]
    dx[1, 0] = th[0]*(1 - X[0]**2)*X[1] - X[0]

    # Jacobian Matrix
    dFdX = np.zeros((n, n))
    dFdX[0, 0] = 0
    dFdX[0, 1] = 1
    dFdX[1, 0] = -2*th[0]*X[0]*X[1]-1
    dFdX[1, 1] = th[0]*(1 - X[0]**2)

    # Parameter gradient
    dFdTh = np.zeros((n, n_theta))
    dFdTh[0, 0] = 0
    dFdTh[1, 0] = (1 - X[0]**2)*X[1]

    return dx, dFdX, dFdTh


def f_obs(X, phi, u, inG):
    # Observation function defining a logistic mapping with fixed scale and slope

    scale = inG["scale"]
    slope = inG["slope"]

    n_phi = 0       # No of Observation Parameters
    nY = 2          # No of Observations
    n = 2           # Model order

    # Observation Equation
    gx = np.zeros((nY, 1))
    gx[0, 0] = scale * 1 / (1 + np.exp(-slope * X[0]))
    gx[1, 0] = scale * 1 / (1 + np.exp(-slope * X[1]))

    # Derivatives of the Observation equation w.r.t
        # - Model states
    dGdX = np.zeros((nY, n))
    dGdX[0, 0] = slope * scale * np.exp(-slope * X[0]) / ((np.exp(-slope * X[0]) + 1)**2)
    dGdX[0, 1] = 0
    dGdX[1, 0] = 0
    dGdX[1, 1] = slope * scale * np.exp(-slope * X[1]) / ((np.exp(-slope * X[1]) + 1)**2)
        # - Observation Parameters
    dGdPhi = np.zeros((nY, n_phi))

    return gx, dGdX, dGdPhi
