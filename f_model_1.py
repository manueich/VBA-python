def f_model(X, th, u, inF):
    # Define Model function with Euler solver

    dt = inF["dt"]

    # Model Equations
    fx = -th[0] * X[0]

    # Next step (Euler)
    fx = fx * dt + X

    # Jacobian
    J = -th[0]

    # Parameter gradient
    H = -X[0]

    return fx, J, H


def f_obs(X):
    # Observation function

    gx = X[0]
    dG_dX = 1
    dG_dP = 0

    return gx, dG_dX, dG_dP
