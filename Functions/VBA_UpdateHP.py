import numpy as np
from Functions import VBA_basics as base


def UpdateHP(data, t, posterior, priors, suffStat, options):

    yd = data["y"]
    td = data["t"]

    a = priors["a"]
    b = priors["b"]

    # Get estimates at current mode
    y, muX, SigmaX, dXdTh, dXdX0, dYdPhi, dYdTh, dYdX0, dG_dP = base.solveODE(t, posterior["muP"], posterior["SigmaP"], data["u"], options)

    # Loop over time series
    for i in range(0, td.size):
        idx = np.where(t == td[0, i])
        gx = y[:, idx[0]]
        iQyt = priors["iQy"][i]

        dy = yd[:, [i]] - gx
        dy2 = dy.T @ iQyt @ dy
        a = a + 0.5*np.size(np.diag(iQyt))
        b = b + 0.5*dy2 + 0.5*np.trace(dG_dP[int(idx[0])] @ iQyt @ dG_dP[int(idx[0])].T @ posterior["SigmaP"])

    # Update Posterior
    posterior.update({"a": a})
    posterior.update({"b": b})

    # Calculate new Free Energy
    F = base.Free_Energy(posterior, priors, suffStat, options)
    Fall = suffStat["F"]
    Fall.append(F)
    suffStat.update({"F": Fall})

    # Get estimates at updated mode
    y, muX, SigmaX, dXdTh, dXdX0, dYdPhi, dYdTh, dYdX0, dG_dP = base.solveODE(t, posterior["muP"], posterior["SigmaP"], data["u"], options)

    # Update suffstat
    model_out = {"y": y,
                 "muX": muX,
                 "SigmaX": SigmaX,
                 "dXdTh": dXdTh,
                 "dXdX0": dXdX0,
                 "dYdPhi": dYdPhi,
                 "dYdTh": dYdTh,
                 "dYdX0": dYdX0}
    suffStat.update({"model_out": model_out})

    return posterior, suffStat