import numpy as np

# Model of directly observed linear damped harmonic oscillator function with the following equation
    # d2X/d2t + 2*th[0]*th[1]*dX/dt + th[0]^2*X = 0

def f_model(X, th, u, inF):

    n = 2               # Model Order
    n_theta = 2         # No of parameters

    # Model Equations
    fx = np.zeros((n, 1))
    fx[0, 0] = X[1]
    fx[1, 0] = -2 * th[0] * th[1] * X[1] - th[0]**2 * X[0]

    # Jacobian Matrix
    J = np.zeros((n, n))
    J[0, 0] = 0
    J[0, 1] = 1
    J[1, 0] = -th[0] ** 2
    J[1, 1] = -2 * th[0] * th[1]

    # Parameter gradient
    H = np.zeros((n, n_theta))
    H[0, 0] = 0
    H[0, 1] = 0
    H[1, 0] = -2 * th[1] * X[1] - 2 * th[0] * X[0]
    H[1, 1] = -2 * th[0] * X[1]

    return fx, J, H


def f_obs(X, phi, u, inG):
    # Observation function
    # y = X[0]

    n_phi = 0       # No of Observation Parameters
    nY = 1          # No of Observations
    n = 2           # Model order

    # Observation Equation
    Y = np.zeros((1, 1))
    Y[0, 0] = X[0]

    # Derivatives of the Observation equation w.r.t
        # - Model states
    dY_dX = np.zeros((nY, n))
    dY_dX[0, 0] = 1
    dY_dX[0, 1] = 0
        # - Observation Parameters
    dY_dPhi = np.zeros((nY, n_phi))

    return Y, dY_dX, dY_dPhi
