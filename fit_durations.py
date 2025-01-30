import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def histogram_log_bins(x, x_min=None, x_max=None, num_bins=20, min_counts=1):
    """
    Generate a histogram with logarithmically spaced bins, normalized by bin width.
    Returns (bin_counts, bin_centers, bin_edges).

    x          : 1D array of data (e.g., avalanche durations).
    x_min      : minimum value for the histogram (if None, uses min of x>0).
    x_max      : maximum value (if None, uses max of x).
    num_bins   : how many log-spaced bins.
    min_counts : bins with fewer than this number of raw counts are dropped.
    """
    # 1) Pick x_min, x_max if not provided
    x = x[x > 0]  # ensure positive durations
    if x_min is None:
        x_min = np.min(x)
    if x_max is None:
        x_max = np.max(x)

    # 2) Logarithmically spaced bin edges
    bin_edges = np.logspace(np.log10(x_min), np.log10(x_max), num_bins + 1)

    # 3) Get raw counts in each bin
    raw_counts, _ = np.histogram(x, bins=bin_edges)
    raw_counts = raw_counts.astype(int)

    # 4) Compute bin widths
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    # 5) Convert raw_counts to density by dividing by bin widths
    density = raw_counts / bin_widths

    # 6) Filter out bins with fewer than min_counts
    keep_mask = (raw_counts >= min_counts)
    density = density[keep_mask]
    kept_edges_left = bin_edges[:-1][keep_mask]
    kept_edges_right = bin_edges[1:][keep_mask]

    # 7) Geometric center for each log bin (common choice)
    bin_centers = np.sqrt(kept_edges_left * kept_edges_right)

    return density, bin_centers, (kept_edges_left, kept_edges_right)

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

def main():
    # ---------------------------------------
    # 1. Read avalanche duration data
    # ---------------------------------------
    df = pd.read_csv("avalanche_durations_first.csv")
    durations = df.values
    durations = durations[durations > 0]  # ensure positivity

    # ---------------------------------------
    # 2. Build a log-binned histogram
    # ---------------------------------------
    counts, bin_centers, edges = histogram_log_bins(
        durations,
        x_min=None,
        x_max=None,
        num_bins=20,
        min_counts=1
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
    lower_cutoff = 2.0
    upper_cutoff = 1e6  # something large
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
    plt.figure(figsize=(8, 5))

    # Plot the binned data
    plt.scatter(bin_centers, counts, s=40, color='blue', label="Log-binned data")

    # Highlight the region we used for the fit
    plt.scatter(x_fit, y_fit, s=60, color='orange', edgecolor='k',
                zorder=3, label="Data used for fit")

    # Overlay the exponential fit
    plt.plot(x_plot, y_plot, 'r--', 
             label=f"Exponential fit\nlambda={lam_fit:.3f} ± {lam_fit_err:.3f}")

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Avalanche duration (log scale)")
    plt.ylabel("Density (counts / bin_width, log scale)")
    plt.title("Avalanche Duration with Log Bins (Log-Log Plot)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
