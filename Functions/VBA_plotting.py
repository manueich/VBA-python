import numpy as np
import matplotlib.pyplot as plt

def plot_data(data):

    ty = data["t"]
    yd = data["y"]

    plt.show()
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
    ax[0, 0].set_title('Model Observations')
    ax[0, 1].set_title('Model States')
    ax[0, 2].set_title('Data vs Model Predictions')

    ax[1, 0].set_title('Observation Parameters')
    ax[1, 1].set_title('Evolution Parameters / Initial Conditions')
    ax[1, 2].set_title('Noise Parameter')

    # for i in range(0, np.shape(yd)[0]):
    #     t = np.reshape(ty, (np.shape(ty)[1]))
    #     y = np.reshape(yd[[i], :], (np.shape(ty)[1]))
    #     ax[0, 0].plot(t, y, marker='o', ls='none', color='b')
    #     plt.pause(1e-17)
    return ax


def plot_model(ax, suffStat, posterior, priors, data, options):

    model_out = suffStat["model_out"]
    t = model_out["t"]
    muX = model_out["muX"]
    SigmaX = model_out["SigmaX"]
    vy = suffStat["vy"]
    gx = suffStat["gx"]

    # Data
    ax[0, 0] = clear_axis(ax[0, 0])
    ax[0, 2] = clear_axis(ax[0, 2])
    for i in range(0, np.shape(gx)[0]):
        yp = np.reshape(gx[[i], :], (np.shape(gx)[1]))
        yd = np.reshape(data["y"][[i], :], (np.shape(data["t"])[1]))
        td = np.reshape(suffStat["data"]["t"], (np.shape(gx)[1]))
        sig = np.sqrt(np.reshape(vy[[i], :], (np.shape(gx)[1])))
        # Data
        ax[0, 0].plot(td, yd, marker='o', ls='none', color='b')
        # Model Pred
        ax[0, 0].plot(td, yp, 'r')
        ax[0, 0].fill_between(td, yp - sig, yp + sig, alpha=0.2, color='r')
        # Data vs Model Pred
        ax[0, 2].plot(yd, yp, marker='o', ls='none', color='b')
        tmp = np.array([np.min(yd), np.max(yd)])
        ax[0, 2].plot(tmp, tmp, color='k')
        plt.pause(1e-17)

    # Model states
    ax[0, 1] = clear_axis(ax[0, 1])
    for i in range(0, np.shape(muX)[0]):
        xp = np.reshape(muX[[i], :], (np.shape(t)[0]))
        sig = np.zeros((t.size))
        for j in range(0, t.size-1):
            sig[j] = np.sqrt(SigmaX[j][i, i])

        ax[0, 1].plot(t, xp, 'g')
        ax[0, 1].fill_between(t, xp - sig, xp + sig, alpha=0.2, color='g')

        plt.pause(1e-17)

    # Observation Parameters
    width = 0.35
    ax[1, 0] = clear_axis(ax[1, 0])
    if options["dim"]["n_phi"] > 0:

        x = np.arange(options["dim"]["n_phi"])+1
        # Priors
        ax[1, 0].bar(x - width/2, priors["muPhi"], width, yerr=np.sqrt(np.diag(priors["SigmaPhi"])), color='r')

        plt.pause(1e-17)

    # Evolution Parameters + Initial Conditions
    ax[1, 1] = clear_axis(ax[1, 1])
    if options["dim"]["n_theta"] > 0:

        # Evolution Paramters
        x = np.arange(options["dim"]["n_theta"])+1
        # Priors
        ax[1, 1].bar(x - width/2, np.reshape(priors["muTheta"], (np.shape(priors["muTheta"])[0])), width,
                     yerr=np.sqrt(np.diag(priors["SigmaTheta"])), color='r')
        # Posterior
        ax[1, 1].bar(x + width/2, np.reshape(posterior["muTheta"], (np.shape(posterior["muTheta"])[0])), width,
                     yerr=np.sqrt(np.diag(posterior["SigmaTheta"])), color='b')

        # Initial Conditions
        x = np.arange(options["dim"]["n"]) + options["dim"]["n_theta"]+2
        # Priors
        ax[1, 1].bar(x - width / 2, np.reshape(priors["muX0"], (np.shape(priors["muX0"])[0])), width,
                     yerr=np.sqrt(np.diag(priors["SigmaTheta"])), color='r')
        # Posterior
        ax[1, 1].bar(x + width / 2, np.reshape(posterior["muX0"], (np.shape(posterior["muX0"])[0])), width,
                     yerr=np.sqrt(np.diag(posterior["SigmaX0"])), color='b')

        plt.pause(1e-17)

    # Noise Parameter
    ax[1, 2] = clear_axis(ax[1, 2])
    # Prior
    ax[1, 2].bar(1 - width / 2, priors["a"]/priors["b"], width, yerr=np.sqrt(priors["a"]/priors["b"]**2), color='r')
    # Posterior
    ax[1, 2].bar(1 + width / 2, posterior["a"]/posterior["b"], width, yerr=np.sqrt(posterior["a"]/posterior["b"]**2), color='b')
    plt.pause(1e-17)

    return ax


def clear_axis(ax):

    ax.lines = []
    ax.collections = []
    ax.patches = []

    return ax
