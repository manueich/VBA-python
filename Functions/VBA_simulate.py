import numpy as np
from Functions import VBA_basics as base
from Functions import VBA_Initialise as Initialise


def simulate(td, t, u, priors, options):
    # Check inputs
    options = Initialise.check_options(options.copy())
    priors = Initialise.check_priors(options, priors.copy())
    data, options, priors = check_data(td, t, u, priors.copy(), options.copy())
    options, priors = check_model(options.copy(), priors.copy(), data)

    # Get initial estimates with priors
    try:
        y, muX, SigmaX = base.solveODE(t, priors["muP"], priors["SigmaP"], data["u"], options)[0:3]
    except:
        raise Exception("The model produces error (see above)")

    if base.isWeird(y):
        raise Exception("Could not simulate model: model generates NaN or Inf!'")

    yd = np.zeros((options["dim"]["nY"], options["dim"]["nD"]))
    epsilon = np.zeros((options["dim"]["nY"], options["dim"]["nD"]))

    for i in range(0, td.size):
        idx = np.where(t == td[0, i])

        sigma = priors["a"] / priors["b"]
        iQyt = priors["iQy"][i]

        C = base.Invert_M(iQyt * sigma)
        epsilon[:, [i]] = np.reshape(np.random.multivariate_normal(np.zeros(options["dim"]["nY"]), C),
                                     (options["dim"]["nY"], 1))

        yd[:, [i]] = y[:, idx[0]] + epsilon[:, [i]]

    return yd


def check_data(ty, t, u, priors, options):
    dim = options["dim"]

    # Check ty
    if np.shape(ty)[0] != 1:
        raise Exception("ty must be a 1 by n array")
    else:
        data = ({"ty": ty})
        dim.update({"nD": np.shape(ty)[1]})

    # Check u
    if not u:
        data.update({"u": np.zeros((1, t.size))})
        dim.update({"nu": 0})
    elif np.shape(u)[1] != np.shape(t)[0]:
        raise Exception("Inputs in u must be specified on the ODE integration time step t")
    elif base.isWeird(u):
        raise Exception("The data in u contains NaNs or Infs")
    else:
        dim.update({"nu": np.shape(u)[0]})

    # Check integration time Grid
    if t[0] != 0:
        raise Exception("The ODE integration time grid must begin with 0")
    dt = options["inF"]["dt"]
    if np.any(np.round(np.diff(t), 8) != dt):
        raise Exception("The ODE integration time grid must match dt in inF")

    options.update({"dim": dim})

    return data, options, priors


def check_model(options, priors, data):
    dim = options["dim"]
    muP = priors["muP"]
    u = data["u"]

    phi = muP[0: dim["n_phi"]]
    th = muP[dim["n_phi"]: dim["n_theta"] + dim["n_phi"]]
    x0 = muP[dim["n_theta"] + dim["n_phi"]: dim["n_theta"] + dim["n_phi"] + dim["n"]]

    # Get functions
    f_model = options["f_model"]
    f_obs = options["f_obs"]

    try:
        x, J, H = f_model(x0, th, u[:, 0], options["inF"])
        y, dY_dX, dY_dPhi = f_obs(x, phi, u[:, 0], options["inG"])
    except:
        raise Exception("The model produces error (see above)")

    if np.shape(x)[0] != dim["n"] or np.shape(x)[1] != 1:
        raise Exception("Model Error: Dimensions of x must be n by 1")

    dim.update({"nY": np.shape(y)[0]})
    if np.shape(y)[1] != 1:
        raise Exception("Model Error: Dimensions of y must be nY by 1")

    options.update({"dim": dim})

    # Check iQy now that we know nY
    if "iQy" in priors:
        if len(priors["iQy"]) != options["dim"]["nD"]:
            raise Exception("The size of iQy must match the given data")
        else:
            iQy = [0] * len(priors["iQy"])
            for i in range(0, len(priors["iQy"])):
                if np.shape(priors["iQy"][i])[0] != options["dim"]["nY"] or np.shape(priors["iQy"][i])[1] != \
                        options["dim"]["nY"]:
                    raise Exception("Inconsistent dimension in iQy")
                else:
                    diQ = np.diag(priors["iQy"][i])
                    iQy[i] = np.diag(diQ) @ priors["iQy"][i] @ np.diag(diQ)
    else:
        iQy = [np.eye(options["dim"]["nY"])]
        for i in range(0, options["dim"]["nD"] - 1):
            iQy.append(np.eye(options["dim"]["nY"]))

    priors.update({"iQy": iQy})

    return options, priors
