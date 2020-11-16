import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl


def plot_data(data):

    ty = data["t"]
    yd = data["y"]

    plt.show()
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.3, hspace=0.3)
    ax[0, 0].set_title('Model Observations')
    ax[0, 1].set_title('Data vs Model Predictions')
    ax[0, 2].set_title('Model States')

    ax[1, 0].set_title('Observation Parameters')
    ax[1, 0].set_ylabel('Prior - Posterior')
    ax[1, 1].set_title('Evolution Parameters / Initial Conditions')
    ax[1, 1].set_ylabel('Prior - Posterior')
    ax[1, 2].set_title('Noise Parameter (log scale)')
    ax[1, 2].set_ylabel('Posterior (log)')

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
    ax[0, 1] = clear_axis(ax[0, 1])
    colors = pl.cm.Set1(np.arange(0, np.shape(gx)[0]))
    maxy = []
    miny = []

    for i in range(0, np.shape(gx)[0]):
        yp = np.reshape(gx[[i], :], (np.shape(gx)[1]))
        yd = np.reshape(data["y"][[i], :], (np.shape(data["t"])[1]))
        td = np.reshape(suffStat["data"]["t"], (np.shape(gx)[1]))
        sig = np.sqrt(np.reshape(vy[[i], :], (np.shape(gx)[1])))
        # Data
        ax[0, 0].plot(td, yd, marker='o', markersize=3, ls='--', color=colors[i])
        maxy.append(np.max(yd))
        miny.append(np.min(yd))
        # Model Pred
        ax[0, 0].plot(td, yp, color=colors[i])
        ax[0, 0].fill_between(td, yp - sig, yp + sig, alpha=0.2, color=colors[i])
        maxy.append(np.max(yp + sig))
        miny.append(np.min(yp - sig))
        # Data vs Model Pred
        ax[0, 1].plot(yd, yp, marker='o', markersize=4, ls='none', color=colors[i])
        tmp = np.array([np.min(yd), np.max(yd)])
        ax[0, 1].plot(tmp, tmp, color='k')
        plt.pause(1e-17)

    ax[0, 0] = rescale_axis(ax[0, 0], maxy, miny)
    ax[0, 1] = rescale_axis(ax[0, 1], [], [])

    # Model states
    ax[0, 2] = clear_axis(ax[0, 2])
    colors = pl.cm.Set1(np.arange(0, np.shape(muX)[0]))
    maxy = []
    miny = []

    for i in range(0, np.shape(muX)[0]):
        yp = np.reshape(muX[[i], :], (np.shape(t)[1]))
        tp = np.reshape(t, (np.shape(t)[1]))
        sig = np.zeros((t.size))
        for j in range(0, t.size-1):
            sig[j] = np.sqrt(SigmaX[j][i, i])

        ax[0, 2].plot(tp, yp, color=colors[i])
        ax[0, 2].fill_between(tp, yp - sig, yp + sig, alpha=0.2, color=colors[i])
        maxy.append(np.max(yp + sig))
        miny.append(np.min(yp - sig))

        plt.pause(1e-17)

    ax[0, 2] = rescale_axis(ax[0, 2], maxy, miny)

    # Observation Parameters
    width = 0.35
    ax[1, 0] = clear_axis(ax[1, 0])
    if options["dim"]["n_phi"] > 0:

        x = np.arange(options["dim"]["n_phi"])+1
        # Priors - Posterior
        ax[1, 0].bar(x, np.reshape(priors["muPhi"] - posterior["muPhi"], (np.shape(posterior["muPhi"])[0])), width,
                     yerr=np.sqrt(np.diag(posterior["SigmaPhi"])), color=colors[0])
        plt.pause(1e-17)
    else:
        ax[1, 0].text(0.4, 0.5, "None")

    # Evolution Parameters + Initial Conditions
    ax[1, 1] = clear_axis(ax[1, 1])
    if options["dim"]["n_theta"] > 0:

        # Evolution Paramters
        x = np.arange(options["dim"]["n_theta"]) + 1
        # Priors - Posterior
        ax[1, 1].bar(x, np.reshape(priors["muTheta"] - posterior["muTheta"], (np.shape(posterior["muTheta"])[0])), width,
                     yerr=np.sqrt(np.diag(posterior["SigmaTheta"])), color=colors[0])
        plt.pause(1e-17)

    # Initial Conditions
    x = np.arange(options["dim"]["n"]) + options["dim"]["n_theta"] + 2
    # Priors - Posterior
    ax[1, 1].bar(x, np.reshape(priors["muX0"] - posterior["muX0"], (np.shape(priors["muX0"])[0])), width,
                 yerr=np.sqrt(np.diag(priors["SigmaX0"])), color=colors[0])
    plt.pause(1e-17)

    # Noise Parameter
    ax[1, 2] = clear_axis(ax[1, 2])
    # Posterior
    ax[1, 2].bar(1, np.log(posterior["a"]/posterior["b"]), width, yerr=np.log(np.sqrt(posterior["a"]/posterior["b"]**2)), color=colors[0])
    plt.pause(1e-17)

    return ax


def clear_axis(ax):

    ax.lines = []
    ax.collections = []
    ax.patches = []
    ax.texts = []

    return ax

def rescale_axis(ax, maxy, miny):

    if maxy:
        buf = max([min(miny), max(maxy)])*0.1
        ax.set_ylim(min(miny)-buf, max(maxy)+buf)
    else:
        ax.relim()
        ax.autoscale_view(tight=False)

    return ax
