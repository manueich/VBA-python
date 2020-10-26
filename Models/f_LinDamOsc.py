import numpy as np

def f_model(X, th, u, inF):
    # Model of linear damped harmonic oscillator function with the following equation
    # d2X/d2t + 2*th[0]*th[1]*dX/dt + th[0]^2*X = 0

    n = 2               # Model Order
    n_theta = 2         # No of parameters
    dt = inF["dt"]

    # Model Equations
    fx = np.zeros((n, 1))
    fx[0] = X[1]
    fx[1] = -2 * th[0] * th[1] * X[1] - th[0]**2 * X[0]

    # Next step (Euler)
    # fx = fx * dt + X

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

    n_phi = 0
    nY = 1
    n = 2

    gx = np.zeros((1, 1))
    gx[0] = X[0]
    dG_dX = np.zeros((nY, n))
    dG_dX[0, 0] = 1
    dG_dX[0, 1] = 0
    dG_dPhi = np.array([[]])

    return gx, dG_dX, dG_dPhi
