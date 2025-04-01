from scipy.optimize import least_squares
import pandas as pd
import numpy as np


def twoPL(x, hill_coeff, ic50):
    a = (1 - 0)
    b = (x / ic50)
    c = 1 + (np.power(b, hill_coeff))
    return (a / c) + 0

def inverse_twoPL(y, hill_coeff, ic50):
    return ic50 * ((((1 - 0) / (y - 0)) - 1) ** (1 / hill_coeff))

def residuals_twoPL(params, y, x):
    """Deviations of data from fitted 4PL curve"""
    hill_coeff, ic50 = params
    err = y - twoPL(x, hill_coeff, ic50)
    return err

def fourPL(x, upper_plato, hill_coeff, ic50, lower_plato):
    a = (upper_plato - lower_plato)
    b = (x / ic50)
    c = 1 + (np.power(b, hill_coeff))
    return (a / c) + lower_plato

def inverse_fourPL(y, upper_plato, hill_coeff, ic50, lower_plato):
    return ic50 * ((((upper_plato - lower_plato) / (y - lower_plato)) - 1) ** (1 / hill_coeff))

def residuals_fourPL(params, y, x):
    """Deviations of data from fitted 4PL curve"""
    upper_plato, hill_coeff, ic50, lower_plato = params
    err = y - fourPL(x, upper_plato, hill_coeff, ic50, lower_plato)
    return err

pl_func = {
    4: fourPL,
    2: twoPL
}

inverse_pl = {
    4: inverse_fourPL,
    2: inverse_twoPL
}

residual_pl = {
    4: residuals_fourPL,
    2: residuals_twoPL,
}

p0 = {
    4: lambda concentrations: [1, 1, np.mean(concentrations), 0],
    2: lambda concentrations: [1, np.mean(concentrations)]
}

bounds = {
    4: [[0, 0, 0, 0], # min
        [1.5, np.inf, np.inf, 1]],
    2: [[0, 0], # min
        [np.inf, np.inf]]
}

def calc_ic_frac(*params, frac, pl=4):
    if len(params) == 4:
        upper_plato, hill_coeff, ic50, lower_plato = params
    if len(params) == 2:
        upper_plato, lower_plato = 1, 0
    curve = lambda x: pl_func[pl](x, *params)
    y = upper_plato * (1 - frac)
    if lower_plato > y:
        return np.inf
    if curve(1e7) > y:
        return np.inf
    try:
        return inverse_pl[pl](y, *params)
    except OverflowError:
        return np.inf

def calc_ic50(
    concentrations, values, pl=4
):
    """
    :param concentrations: list of concentrations
    :param values: list of values normalized to the control, i.e. for low concentrations values are close to 1
    :return: ic50
    """
    if type(concentrations) is pd.Series:
        concentrations = concentrations.values

    if type(values) is pd.Series:
        values = values.values
    
    r_squared, ic50_curve,  = [None] * 2
    valid_fit = True
    try:
        plsq = least_squares(
            residual_pl[pl], 
            p0[pl](concentrations), 
            args=(values, concentrations), 
            bounds=bounds[pl],
            max_nfev=1000 * 846
        )
        valid_fit = plsq.success
        message = plsq.message
    except Exception as e:
        valid_fit = False
        message = str(e)

    if not valid_fit:
       return valid_fit, {"message": message} 

    if pl == 4:
        upper_plato, hill_coeff, ic50, lower_plato = plsq.x
        message = f"Lower Plato:{lower_plato:.2f}, Upper Plato:{upper_plato:.2f}"
    if pl == 2:
        hill_coeff, ic50= plsq.x
        upper_plato = 1
        lower_plato = 0
        message = ''
    
    def ic50_curve(x):
        return pl_func[pl](x, *plsq.x)

    residuals = values - ic50_curve(concentrations)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((values - np.mean(values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    ret_object = {
        "message": message,
    }

    if ic50 is not None:
        ret_object["ic50"] = ic50
        # ret_object['curve'] = ic50_curve
        ret_object["r_squared"] = r_squared
        ret_object["mae"] = np.abs(residuals).mean()
        ret_object["lower_plato"] = lower_plato
        ret_object["upper_plato"] = upper_plato
        ret_object["hill_coeff"] = hill_coeff

    return valid_fit, ret_object