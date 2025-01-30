import numpy as np
from scipy.stats import kurtosis, skew
from scipy.ndimage import label
import pywt

__all__ = ['filter_wavelet_coefficients_paper', 'tune_threshold', 'extract_avalanches', 'compute_pdf']

def filter_wavelet_coefficients_paper(coeffs, C):
    """
    Filters wavelet coefficients based on the method described in the paper.
    Coefficients are kept only if their square is less than C times the mean squared value.
    """
    filtered_coeffs = []
    for c in coeffs:
        # Compute the mean squared value of coefficients at this scale
        mean_squared = np.mean(c ** 2)
        
        # Apply the paper's thresholding condition
        filtered_c = np.where(c**2 < C * mean_squared, c, 0)
        filtered_coeffs.append(filtered_c)
    
    return filtered_coeffs

def tune_threshold(log_returns, wavelet='db1', level=4, target_kurtosis=3, target_skew=0, tolerance=0.1, max_iter = 1000):
    """
    Dynamically tune the threshold parameter C for wavelet filtering based on kurtosis and skewness. 
    The parameter C is a threshold coefficient that can be tuned such that Gaussian noise is filtered. 
    """
    coeffs = pywt.wavedec(log_returns, wavelet=wavelet, level=level)
    C = 1.0

    for _ in range(max_iter):
        # Filter coefficients using the current C
        filtered_coeffs = filter_wavelet_coefficients_paper(coeffs, C)
        filtered_signal = pywt.waverec(filtered_coeffs, wavelet=wavelet, mode='periodization')
        # Truncate to match original length:
        filtered_signal = filtered_signal[:len(log_returns)]

        # Compute residuals and their kurtosis/skewness
        residual_signal = log_returns - filtered_signal
        residual_kurtosis = kurtosis(residual_signal, fisher=False)
        residual_skew = skew(residual_signal)

        # Check if kurtosis and skewness are within the tolerance
        if (
            abs(residual_kurtosis - target_kurtosis) < tolerance
            and abs(residual_skew - target_skew) < tolerance
        ):
            break

        # Increment C to filter more aggressively
        C += 0.1
    #print(coeffs)
    return C, filtered_signal, residual_signal

def extract_avalanches(residual_signal, avalanche_threshold=0.01):
    """
    Extracts avalanches from the residual signal based on a threshold.
    Returns the avalanche sizes and durations.
    """
    labeled_array, num_features = label(np.abs(residual_signal) > avalanche_threshold)
    avalanche_sizes = []
    avalanche_durations = []
    avalanche_intertimes = []
    last_avalanche_time = 0
    for i in range(1, num_features + 1):
        indices = np.where(labeled_array == i)[0]
        size = np.sum(np.abs(residual_signal[indices]))  # Sum of residuals during the avalanche
        duration = len(indices)  # Number of time steps in the avalanche
        if last_avalanche_time != 0:
            intertime = indices[0] - last_avalanche_time
            avalanche_intertimes.append(intertime)
        last_avalanche_time = indices[-1]
        avalanche_sizes.append(size)
        avalanche_durations.append(duration)
    return avalanche_sizes, avalanche_durations

# Compute PDF for original and filtered log returns
def compute_pdf(data, bins=50):
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist