import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def histogram_log_bins(x, x_min=None, x_max=None, num_bins=20, min_hits=1):
    """
    Generate a histogram with logarithmically spaced bins, normalized by bin width.
    Returns (bin_counts, bin_centers, total_hits).

    Arguments:
    ----------
    x         : 1D array of data (e.g., avalanche sizes)
    x_min     : minimum value for the histogram (if None, use np.min(x))
    x_max     : maximum value for the histogram (if None, use np.max(x))
    num_bins  : how many bins to use (logarithmically spaced)
    min_hits  : bins with fewer than this number of raw (integer) counts are excluded

    The returned bin_counts are normalized by bin width, so it's effectively
    a 'density' histogram (counts per unit x).
    """
    # 1) Determine min/max if not provided
    if x_min is None:
        x_min = np.min(x[x>0])  # ensure we only use positive values for log bins
    if x_max is None:
        x_max = np.max(x)

    # 2) Logarithmically spaced bin edges
    bin_edges = np.logspace(np.log10(x_min), np.log10(x_max), num_bins + 1)

    # 3) Raw histogram counts (integer)
    raw_counts, _ = np.histogram(x, bins=bin_edges)
    raw_counts = raw_counts.astype(int)

    # 4) Optional: filter out bins with fewer than min_hits
    #    We'll build a mask that we apply *after* normalizing so we keep bin centers aligned.
    keep_mask = (raw_counts >= min_hits)

    # 5) Convert raw counts to 'density' by dividing by bin width
    #    bin_width[i] = bin_edges[i+1] - bin_edges[i]
    bin_counts = raw_counts.astype(float)
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    bin_counts = bin_counts / bin_widths

    # 6) Total hits (just sum of raw counts)
    total_hits = np.sum(raw_counts)

    # 7) For log-spaced bins, a common approach is to take the bin center
    #    as the geometric mean of edges: sqrt(edge[i] * edge[i+1])
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    # 8) Apply the keep_mask for bins that had enough counts
    bin_counts = bin_counts[keep_mask]
    bin_centers = bin_centers[keep_mask]

    return bin_counts, bin_centers, total_hits

def main():
    # ------------------------------------------------------------------------
    # A) MERGE AVALANCHE-SIZE DATA FROM TWO CSV FILES
    # ------------------------------------------------------------------------
    #  Adjust to your actual filenames/column names
    df1 = pd.read_csv("avalanche_durations_first.csv")        # or header=None, if no headers
    df2 = pd.read_csv("avalanche_intertimes_second.csv")  # or header=None

    # If these files each have a column named "avalanche_size", do:
    avalanche_sizes_1 = df1.values
    avalanche_sizes_2 = df2.values

    # Merge (append) them into one array:
    avalanche_sizes = np.concatenate([avalanche_sizes_1, avalanche_sizes_2])

    # Optional: filter out zero/negative values if that makes no physical sense:
    avalanche_sizes = avalanche_sizes[avalanche_sizes > 0]

    # ------------------------------------------------------------------------
    # B) BUILD THE LOGARITHMIC HISTOGRAM
    # ------------------------------------------------------------------------
    # Adjust parameters as needed
    counts, bin_centers, total_hits = histogram_log_bins(
        x=avalanche_sizes,
        x_min=None,      # or a specific positive float
        x_max=None,      # or a specific float < max of data
        num_bins=20,
        min_hits=1
    )

    # Remove any zero counts or zero bin-centers to avoid log(0)
    valid_mask = (counts > 0) & (bin_centers > 0)
    counts = counts[valid_mask]
    bin_centers = bin_centers[valid_mask]
    # ------------------------------------------------------------------------
    # C) CONVERT TO LOG-LOG SPACE
    # ------------------------------------------------------------------------
    log_x = np.log10(bin_centers)
    log_y = np.log10(counts)

    # ------------------------------------------------------------------------
    # D) DEFINE LOWER AND UPPER CUTOFFS FOR FITTING
    # ------------------------------------------------------------------------
    #  We only fit the region x_cutoff_lower <= x <= x_cutoff_upper
    x_cutoff_lower = .08
    x_cutoff_upper = 6000.0

    fit_mask = (bin_centers >= x_cutoff_lower) & (bin_centers <= x_cutoff_upper)
    fit_log_x = log_x[fit_mask]
    fit_log_y = log_y[fit_mask]

    # ------------------------------------------------------------------------
    # E) LINEAR FIT IN LOG-LOG SPACE WITH ERROR ESTIMATES
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
    # F) GENERATE SMOOTH LINE FOR PLOTTING THE FIT
    # ------------------------------------------------------------------------
    log_x_min = fit_log_x.min()
    log_x_max = fit_log_x.max()
    x_fit_line = np.logspace(log_x_min, log_x_max, 200)
    y_fit_line = 10 ** (m * np.log10(x_fit_line) + c)

    # ------------------------------------------------------------------------
    # G) PLOT THE DATA & THE FIT
    # ------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))

    # 1) Plot all data (log-binned)
    plt.scatter(
        bin_centers, counts,
        color='blue', s=30, label='Log-binned Data'
    )

    # 2) Highlight the portion used for fitting
    plt.scatter(
        bin_centers[fit_mask], counts[fit_mask],
        color='orange', edgecolors='k', s=60, zorder=3, label='Points used for fit'
    )

    # 3) Plot the best-fit line
    label_fit = (
        f'alpha = {alpha:.3f}±{alpha_err:.3f}'
    )
    plt.plot(x_fit_line, y_fit_line, 'r--', label=label_fit)

    # Log scales
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Time Between Avalanches (timesteps)")
    plt.ylabel("Density (counts / bin width)")
    plt.title("Log-Binned Avalanche In-Between Times")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------------
    # H) PRINT OUT THE FIT RESULTS
    # ------------------------------------------------------------------------
    print("="*60)
    print(" FIT RESULTS")
    print("="*60)
    print(f"Slope (m)          = {m:.3f} ± {m_err:.3f}")
    print(f"Intercept (c)      = {c:.3f} ± {c_err:.3f}  (log10 scale)")
    print(f"Alpha (=-m)        = {alpha:.3f} ± {alpha_err:.3f}")
    print(f"Total hits (all bins) = {total_hits}")
    print("="*60)


if __name__ == "__main__":
    main()
