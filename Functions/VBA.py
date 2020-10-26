import numpy as np
from Functions import VBA_UpdateHP as VBA_HP, VBA_GaussNewton as VBA_GN, VBA_Initialise as VBA_init, \
    VBA_plotting as plotting
from timeit import default_timer as timer
import matplotlib.pyplot as plt


def main(data, t, priors, options):

    # Check inputs Get initial estimates with priors
    posterior, priors, options, suffStat = VBA_init.Initialize(data, t, priors.copy(), options.copy())

    # Plot data and priors
    if options["Display"]:
        ax = plotting.plot_data(data)
        ax = plotting.plot_model(ax, suffStat, posterior, priors, data, options)

    # -----------------------------------------------------------
    # Main iteration loop maximising Free Energy
    stop = False
    it = 0
    start = timer()
    if options["verbose"]:
        print("Maximising Free Energy ...")
    while not stop:
        it = it + 1
        F0 = suffStat["F"][-1]

        # Update Evolution parameters
        posterior, suffStat = VBA_GN.GaussNewton(data, t, posterior.copy(), priors, suffStat.copy(), options)

        # Update Noise parameters
        if options["updateHP"]:
            posterior, suffStat = VBA_HP.UpdateHP(data, t, posterior.copy(), priors, suffStat.copy(), options)

        # Display progress
        if options["verbose"]:
            dF = suffStat["F"][-1] - F0
            print('VB iteration #', it, '         F=', np.round(float(suffStat["F"][-1]), 1), '         ... dF=', np.round(float(dF), 3))

        # Plot current State
        if options["Display"]:
            ax = plotting.plot_model(ax, suffStat, posterior, priors, data, options)

        # Check Convergence Criterion
        F = suffStat["F"]
        dF = F[-1] - F[-2]

        if it == options["MaxIter"] or np.abs(dF) <= options["TolFun"]:
            stop = True
            if np.abs(dF) <= options["TolFun"]:
                conv = 1
            else:
                conv = 0
    end = timer()
    posterior.update({"F": F[-1]})
    posterior.update({"ModelOut": suffStat["model_out"]})

    out = {"it": it,
           "F": suffStat["F"]}

    print('VB inversion complete (took ~', end-start, 's).')

    # Keep Figure Window displayed
    if options["Display"]:
        plt.show()

    return posterior, out

# Notes

    # - Plotting
    # - Fit metrics
    # - Static models


