import numpy as np
from scipy.interpolate import UnivariateSpline
from numpy.polynomial.polynomial import Polynomial

def smooth(scalars, weight):
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - np.power(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)
    
    return np.array(smoothed)

def smooth_spline(x, y, s=10):
    """
    Smooths the data using spline interpolation.
    Args:
    x (list or numpy array): The independent variable.
    y (list or numpy array): The dependent variable to smooth.
    s (float): Smoothing factor for spline. Increase for more smoothing.

    Returns:
    function: A spline function that can be called with new x values.
    """
    spline = UnivariateSpline(x, y, s=s)
    return spline

def moving_average(data, window_size, mode='same'):
    """
    Smooths the data using a simple moving average.
    Args:
    data (list or numpy array): The input data to smooth.
    window_size (int): The number of points to include in the moving average window.

    Returns:
    numpy array: Smoothed data.
    """
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode=mode)

def smooth_ema(data, alpha):
    ema = [data[0]]
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[i-1])
    return np.array(ema)

def smooth_polynomial(data, degree):
    x = np.arange(len(data))
    coefs = Polynomial.fit(x, data, deg=degree)
    return coefs(x)