import numpy as np
from scipy.stats import linregress

'''
    Ideal FeD model.
'''

# Circuit Parameters
RSENSE = 1e8        # Sense resistor (Ohms)
SCALE1 = 1          # First voltage scaling factor
OFFSET1 = 0         # First voltage offset (V)
A_LOG = -0.02625305 # Logarithmic amplifier coefficient A
B_LOG = 0.00055252  # Logarithmic amplifier coefficient B
C_LOG = -0.19721978 # Logarithmic amplifier coefficient C

# Final scaling parameters
SCALE2_EXP = -5.262585    # Experimental final voltage scaling factor
OFFSET2_EXP = 0.215121    # Experimental final voltage offset (V)
SCALE2_IDEAL = -4.376906  # Ideal final voltage scaling factor
OFFSET2_IDEAL = 0.385988  # Ideal final voltage offset (V)

# Voltage ranges
VIN_MIN = 0.0        # Minimum input voltage for standard operation (V)
VIN_MAX = 3.3        # Maximum input voltage for standard operation (V)
VIN_POINTS = 100     # Number of points for standard operation

# Extended voltage range for linear extrapolation
VIN_MIN_EXTENDED = 0.0      # Extended minimum voltage (V)
VIN_MAX_EXTENDED = 8.0      # Extended maximum voltage (V)
VIN_POINTS_EXTENDED = 800   # Points for smooth extrapolation curves

# Validated experimental range (where we fit the linear models)
VIN_MIN_VALID = 6.45        # Start of valid range (V)
VIN_MAX_VALID = 7.45        # End of valid range (V)
VIN_POINTS_VALID = 100      # Points within valid range for fitting

# First stage transfer function parameters
FIRST_STAGE_SLOPE = 0.3030303    # Slope of first stage
FIRST_STAGE_OFFSET = 6.45        # Offset of first stage

# Experimental fitted parameters
SUBSET_INDICES = [1, 2, 3, 4, 5, 6, 7, 8]
REAL_STATES = [15, 16, 18, 19, 20, 21, 22, 23]
REAL_A_COEFFICIENTS = [1.341, 1.3, 1.226, 1.187, 1.129, 1.055, 1.007, 0.897]
REAL_G_VALUES = [5.57e-13, 5.29e-13, 6.17e-13, 4.16e-13, 3.7e-13, 5.19e-13, 6.46e-13, 1.14e-12]

# Ideal parameters (perfect linear progression and uniform G)
IDEAL_A_COEFFICIENTS = np.linspace(REAL_A_COEFFICIENTS[0], REAL_A_COEFFICIENTS[-1], 8).tolist()
IDEAL_G_VALUES = [np.mean(REAL_G_VALUES)] * 8

def calculate_circuit_response_standard(vin, G, A, scale2, offset2):
    """
    Calculate circuit response for standard 0-3.3V operation with first stage transformation
    """
    # Step 1: Apply first stage transfer function
    Vin_transformed = FIRST_STAGE_SLOPE * vin + FIRST_STAGE_OFFSET

    # Step 2: Calculate current from exponential fit
    I = G * np.exp(A * Vin_transformed)

    # Step 3: Calculate sense voltage
    Vsense = RSENSE * I

    # Step 4: Calculate first scaled voltage
    Vin_log = SCALE1 * Vsense + OFFSET1

    # Step 5: Calculate logarithmic amplifier output
    log_arg = np.maximum(Vin_log + B_LOG, 1e-12)
    Vout_log = A_LOG * np.log(log_arg) + C_LOG

    # Step 6: Calculate final output voltage
    Vout = scale2 * Vout_log + offset2

    return Vout, I, Vsense, Vin_log, Vout_log, Vin_transformed

def calculate_circuit_response_direct(vin, G, A):
    """
    Calculate circuit response for direct voltage input (for extrapolation)
    """
    # Step 1: Calculate current from exponential fit (no first stage transformation)
    I = G * np.exp(A * vin)

    # Step 2: Calculate sense voltage
    Vsense = RSENSE * I

    # Step 3: Calculate first scaled voltage
    Vin_log = SCALE1 * Vsense + OFFSET1

    # Step 4: Calculate logarithmic amplifier output
    log_arg = np.maximum(Vin_log + B_LOG, 1e-12)
    Vout_log = A_LOG * np.log(log_arg) + C_LOG

    # Step 5: Calculate final output voltage (using ideal scaling for extrapolation)
    Vout = SCALE2_IDEAL * Vout_log + OFFSET2_IDEAL

    return Vout, I, Vsense, Vin_log, Vout_log

def generate_experimental_data():
    """Generate experimental circuit response data (0-3.3V)"""
    vin_array = np.linspace(VIN_MIN, VIN_MAX, VIN_POINTS)

    experimental_data = {
        'vin_data': [],
        'vout_data': [],
        'states': REAL_STATES.copy()
    }

    for i in range(len(SUBSET_INDICES)):
        G = REAL_G_VALUES[i]
        A = REAL_A_COEFFICIENTS[i]

        vout, _, _, _, _, _ = calculate_circuit_response_standard(vin_array, G, A, SCALE2_EXP, OFFSET2_EXP)

        experimental_data['vin_data'].append(vin_array)
        experimental_data['vout_data'].append(vout)

    return experimental_data

def generate_ideal_data():
    """Generate ideal circuit response data (0-3.3V)"""
    vin_array = np.linspace(VIN_MIN, VIN_MAX, VIN_POINTS)

    ideal_data = {
        'vin_data': [],
        'vout_data': [],
        'states': REAL_STATES.copy()
    }

    for i in range(len(SUBSET_INDICES)):
        G = IDEAL_G_VALUES[i]
        A = IDEAL_A_COEFFICIENTS[i]

        vout, _, _, _, _, _ = calculate_circuit_response_standard(vin_array, G, A, SCALE2_IDEAL, OFFSET2_IDEAL)

        ideal_data['vin_data'].append(vin_array)
        ideal_data['vout_data'].append(vout)

    return ideal_data

def generate_extrapolated_data():
    """Generate extrapolated ideal data with linear fitting"""
    # Voltage arrays
    vin_valid = np.linspace(VIN_MIN_VALID, VIN_MAX_VALID, VIN_POINTS_VALID)
    vin_extended = np.linspace(VIN_MIN_EXTENDED, VIN_MAX_EXTENDED, VIN_POINTS_EXTENDED)

    extrapolated_data = {
        'vin_extended': vin_extended,
        'vin_valid': vin_valid,
        'vout_extended': [],
        'vout_valid': [],
        'states': REAL_STATES.copy()
    }

    for i in range(len(SUBSET_INDICES)):
        G = IDEAL_G_VALUES[i]
        A = IDEAL_A_COEFFICIENTS[i]

        # Calculate circuit response in valid range only
        vout_valid, _, _, _, _ = calculate_circuit_response_direct(vin_valid, G, A)

        # Fit linear model to valid range data
        slope, intercept, _, _, _ = linregress(vin_valid, vout_valid)

        # Calculate linear extrapolation over extended range
        vout_extended_linear = slope * vin_extended + intercept

        extrapolated_data['vout_valid'].append(vout_valid)
        extrapolated_data['vout_extended'].append(vout_extended_linear)

    return extrapolated_data

