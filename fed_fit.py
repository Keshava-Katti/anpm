import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit


'''
    Fits exponential function to experimental FeD I-V data.
'''

# Data path
DATA_PATH = '../Figures/States/'

# Selected states and parameters (from your optimization)
SELECTED_STATES = [15, 16, 18, 19, 20, 21, 22, 23]  # 8 selected states
ALL_STATES = list(range(1, 28))  # All 27 states
NON_SELECTED_STATES = [s for s in ALL_STATES if s not in SELECTED_STATES]

# Voltage range for optimization
V_MIN_HIGHLIGHT = 6.45  # V
V_MAX_HIGHLIGHT = 7.45  # V

# Experimental fitted parameters (from your optimization results)
REAL_A_COEFFICIENTS = [1.341, 1.3, 1.226, 1.187, 1.129, 1.055, 1.007, 0.897]
REAL_G_VALUES = [5.57e-13, 5.29e-13, 6.17e-13, 4.16e-13, 3.7e-13, 5.19e-13, 6.46e-13, 1.14e-12]

# Ideal parameters (perfect linear progression and uniform G)
SUBSET_INDICES = [1, 2, 3, 4, 5, 6, 7, 8]
IDEAL_A_COEFFICIENTS = np.linspace(REAL_A_COEFFICIENTS[0], REAL_A_COEFFICIENTS[-1], 8).tolist()
IDEAL_G_VALUES = [np.mean(REAL_G_VALUES)] * 8  # Uniform G values

def load_iv_data(state_num):
    """Load IV data for a specific state"""
    filename = f"State{state_num}.csv"
    filepath = DATA_PATH + filename

    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        return df['V'].values, df['I'].values
    except Exception as e:
        print(f"Error loading State {state_num}: {e}")
        return None, None

def exponential_func(V, G, alpha):
    """Exponential function: I = G * exp(alpha * V)"""
    return G * np.exp(alpha * V)

def fit_exponential(V, I):
    """Fit exponential function to IV data"""
    # Filter positive currents and finite values
    mask = (I > 0) & np.isfinite(I) & np.isfinite(V)
    if np.sum(mask) < 3:
        return None, None, 0

    V_fit = V[mask]
    I_fit = I[mask]

    try:
        # Use linear regression on log(I) vs V for initial guess
        log_I = np.log(I_fit)
        slope, intercept, r_value, _, _ = linregress(V_fit, log_I)

        G_initial = np.exp(intercept)
        alpha_initial = slope

        # Nonlinear fit
        popt, _ = curve_fit(exponential_func, V_fit, I_fit,
                           p0=[G_initial, alpha_initial], maxfev=2000)

        G, alpha = popt

        # Calculate R²
        I_pred = exponential_func(V_fit, G, alpha)
        ss_res = np.sum((I_fit - I_pred) ** 2)
        ss_tot = np.sum((I_fit - np.mean(I_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return G, alpha, r_squared

    except Exception as e:
        print(f"Fitting failed: {e}")
        return None, None, 0

    