import numpy as np

def f_model(X, th, u, inF):
    # Define Model function with Euler solver

    n = 2
    pn = 3
    dt = inF["dt"]

    # Model Equations
    fx = np.zeros((n, 1))
    fx[0] = X[1]
    fx[1] = -2 * th[0] * th[1] * X[1] - th[0]**2 * X[0] - th[2]

    # Next step (Euler)
    fx = fx * dt + X

    # Jacobian
    J = np.zeros((n, n))
    J[0, 0] = 0
    J[0, 1] = 1
    J[1, 0] = -th[0] ** 2
    J[1, 1] = -2 * th[0] * th[1]

    # Parameter gradient
    H = np.zeros((n, pn))
    H[0, 0] = 0
    H[0, 1] = 0
    H[0, 2] = 0
    H[1, 0] = -2 * th[1] * X[1] - 2 * th[0] * X[0]
    H[1, 1] = -2 * th[0] * X[1]
    H[1, 2] = -1

    return fx, J, H


def f_obs(X, phi, u, inG):
    # Observation function

    n_phi = 1
    nY = 1
    n = 2

    # gx = np.zeros((nY, 1))
    # gx[0] = X[0] * phi[0]
    # gx[1] = X[1]
    # dG_dX = np.zeros((nY, n))
    # dG_dX[0, 0] = phi[0]
    # dG_dX[0, 1] = 0
    # dG_dX[1, 0] = 0
    # dG_dX[1, 1] = 1
    # dG_dPhi = np.zeros((nY, n_phi))
    # dG_dPhi[0, 0] = X[0]
    # dG_dPhi[1, 0] = 0

    gx = np.zeros((1, 1))
    gx[0] = X[0]
    dG_dX = np.zeros((nY, n))
    dG_dX[0, 0] = 1
    dG_dX[0, 1] = 0
    dG_dPhi = np.array([[]])

    # gx = phi[0]
    # dG_dX = np.array([[]])
    # dG_dPhi = np.zeros((nY, n_phi))
    # dG_dPhi[0, 0] = 1

    return gx, dG_dX, dG_dPhi
