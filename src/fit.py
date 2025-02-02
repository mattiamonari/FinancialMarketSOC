import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plots import histogram_log_bins, plot_curve_power_law, plot_curve_exponential


__all__ = ["fit_curve_exponential", "fit_curve_power_law"]

def fit_exponential_semilog(xdata, ydata):
    """
    Fit y ~ exp(intercept + slope*x), i.e. ln(y) = intercept + slope*x.
    Returns (lambda, intercept, slope).
    slope = -lambda, so lambda = -slope.
    """
    # We'll fit ln(y) = slope * xdata + intercept.
    log_y = np.log(ydata)
    p, cov = np.polyfit(xdata, log_y, deg=1, cov=True)
    slope, intercept = p
    slope_err = np.sqrt(cov[0, 0])
    intercept_err = np.sqrt(cov[1, 1])

    lam = -slope
    lam_err = slope_err
    return lam, lam_err, intercept, intercept_err

def fit_exponential_mle(xdata):
    """
    Simple maximum-likelihood estimate for an exponential distribution:
       lambda_hat = 1 / mean(x).
    Returns lam, standard error, plus additional info.
    """
    # If the data truly follows Exp(lambda), the MLE is 1 / mean(x).
    lam = 1.0 / np.mean(xdata)
    # Approx standard error for lam (assuming large sample):
    #    var(lambda_hat) ~ lam^2 / N
    # => std = lam / sqrt(N)
    lam_err = lam / np.sqrt(len(xdata))
    return lam, lam_err

def fit_curve_exponential(df, lower_cutoff, upper_cutoff, xlabel, title, name, 
                          num_of_bins=20, min_hits=1):

    durations = df.values
    # ---------------------------------------
    # 2. Build a log-binned histogram
    # ---------------------------------------
    counts, bin_centers, edges = histogram_log_bins(
        durations,
        x_min=None,
        x_max=None,
        num_of_bins=num_of_bins,
        min_hits=min_hits
    )
    
    # Filter out zero counts if they remain
    mask_pos = (counts > 0)
    counts = counts[mask_pos]
    bin_centers = bin_centers[mask_pos]

    # ---------------------------------------
    # 3. Fit the exponential tail
    #    Option A: Semilog fit on the bin-centers
    # ---------------------------------------
    # We'll do ln(counts) = intercept + slope * (bin_center),
    # slope = -lambda => lambda = -slope
    # But keep in mind these are "binned" data, so it's approximate.
    
    # Possibly define a cutoff if you only want the tail:
    mask_fit = (bin_centers >= lower_cutoff) & (bin_centers <= upper_cutoff)
    
    x_fit = bin_centers[mask_fit]
    y_fit = counts[mask_fit]
    
    lam_fit, lam_fit_err, intercept, intercept_err = fit_exponential_semilog(x_fit, y_fit)
    print("[Semilog Fit on Binned Data]")
    print(f"  lambda = {lam_fit:.4f} ± {lam_fit_err:.4f}")
    print(f"  intercept = {intercept:.4f} ± {intercept_err:.4f}")
    
    # ---------------------------------------
    # 4. (Optional) MLE approach on raw data (often more accurate)
    # ---------------------------------------
    lam_mle, lam_mle_err = fit_exponential_mle(durations[durations >= lower_cutoff])
    print("[MLE on Raw Data]")
    print(f"  lambda = {lam_mle:.4f} ± {lam_mle_err:.4f}")

    # ---------------------------------------
    # 5. Generate the best-fit exponential for plotting
    #    We'll use the semilog-fit result for demonstration
    # ---------------------------------------
    # y = exp(intercept + slope*x)
    # slope = -lam_fit
    slope = -lam_fit
    
    # We'll just plot from the minimal x_fit to the maximal
    x_plot = np.logspace(np.log10(x_fit.min()), np.log10(x_fit.max()), 200)
    y_plot = np.exp(intercept + slope * x_plot)

    # ---------------------------------------
    # 6. Plot in log–log space
    #    Even though it's an exponential, we can do it
    # ---------------------------------------
    plot_curve_exponential(x_fit, y_fit, x_plot, y_plot, bin_centers, counts, 
                           lam_fit, lam_fit_err, xlabel, title, savefig=True, 
                           name=name)


def fit_curve_power_law(df, lower_cutoff, upper_cutoff, xlabel, title, name,
                        num_of_bins=20, min_hits=1):
    # ------------------------------------------------------------------------
    # A) BUILD THE LOGARITHMIC HISTOGRAM
    # ------------------------------------------------------------------------
    # Adjust parameters as needed
    counts, bin_centers, total_hits = histogram_log_bins(
        x=df,
        x_min=None,      # or a specific positive float
        x_max=None,      # or a specific float < max of data
        num_of_bins=num_of_bins,
        min_hits=min_hits
    )

    # Remove any zero counts or zero bin-centers to avoid log(0)
    valid_mask = (counts > 0) & (bin_centers > 0)
    counts = counts[valid_mask]
    bin_centers = bin_centers[valid_mask]
    # ------------------------------------------------------------------------
    # B) CONVERT TO LOG-LOG SPACE
    # ------------------------------------------------------------------------
    log_x = np.log10(bin_centers)
    log_y = np.log10(counts)

    # ------------------------------------------------------------------------
    # C) DEFINE LOWER AND UPPER CUTOFFS FOR FITTING
    # ------------------------------------------------------------------------
    #  We only fit the region x_cutoff_lower <= x <= x_cutoff_upper

    fit_mask = (bin_centers >= lower_cutoff) & (bin_centers <= upper_cutoff)
    fit_log_x = log_x[fit_mask]
    fit_log_y = log_y[fit_mask]

    # ------------------------------------------------------------------------
    # D) LINEAR FIT IN LOG-LOG SPACE WITH ERROR ESTIMATES
    #    np.polyfit(..., cov=True) -> returns (coeffs, covariance_matrix)
    # ------------------------------------------------------------------------
    p, cov = np.polyfit(fit_log_x, fit_log_y, deg=1, cov=True)
    m, c = p  # slope, intercept

    # The diagonal of 'cov' are variances -> standard errors
    m_err = np.sqrt(cov[0, 0])
    c_err = np.sqrt(cov[1, 1])

    # For a power-law y ~ x^(-alpha), slope m = -alpha -> alpha = -m
    alpha = -m
    alpha_err = m_err  # same magnitude, just negative sign

    # ------------------------------------------------------------------------
    # E) GENERATE SMOOTH LINE FOR PLOTTING THE FIT
    # ------------------------------------------------------------------------
    log_x_min = fit_log_x.min()
    log_x_max = fit_log_x.max()
    x_fit_line = np.logspace(log_x_min, log_x_max, 200)
    y_fit_line = 10 ** (m * np.log10(x_fit_line) + c)

    # ------------------------------------------------------------------------
    # F) PLOT THE DATA & THE FIT
    # ------------------------------------------------------------------------
    plot_curve_power_law(x_fit_line, y_fit_line, bin_centers, counts, fit_mask, 
                         alpha, alpha_err, xlabel, title, savefig=True, 
                         name=name)
    

    # ------------------------------------------------------------------------
    # G) PRINT OUT THE FIT RESULTS
    # ------------------------------------------------------------------------
    print("="*60)
    print(" FIT RESULTS")
    print("="*60)
    print(f"Slope (m)          = {m:.3f} ± {m_err:.3f}")
    print(f"Intercept (c)      = {c:.3f} ± {c_err:.3f}  (log10 scale)")
    print(f"Alpha (=-m)        = {alpha:.3f} ± {alpha_err:.3f}")
    print(f"Total hits (all bins) = {total_hits}")
    print("="*60)


