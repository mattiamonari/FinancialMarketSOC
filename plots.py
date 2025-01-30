import matplotlib.pyplot as plt
import numpy as np
from wavelet import compute_pdf
from scipy.stats import norm
import powerlaw

__all__ = ['plot_returns', 'plot_market_price', 'plot_ratio_buyers_sellers', 'plot_weighted_volumes', 'plot_returns_vs_time', 'plot_returns_distribution', 'plot_original_vs_filtered_log_returns_pdf', 'plot_avalanches_on_log_returns', 'plot_avalanche_sizes']

def plot_returns_vs_time(returns, saveFig=False):
    plt.plot(returns)
    plt.xlabel('Time Steps')
    plt.ylabel('Returns')
    plt.title('Price Returns')
    
    if saveFig:
        plt.savefig('returns_vs_time.pdf')
    else:
        plt.show()

def plot_returns_distribution(returns, saveFig=False):
    plt.hist(returns, bins=100)
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.title('Distribution of Returns')
    
    if saveFig:
        plt.savefig('returns_distribution.pdf')
    else:
        plt.show()

def plot_returns(returns, saveFig=False):
    plot_returns_vs_time(returns, saveFig)
    plot_returns_distribution(returns, saveFig)

def plot_market_price(prices, moving_avg, profiler_view=False, saveFig=False):
    if profiler_view:
        plt.subplot(3, 1, 1)

    plt.plot(prices, label='Market Price', color='blue')
    plt.plot(moving_avg, label='Moving Average', color='red')
    plt.xlabel('Time Steps')
    plt.ylabel('Market Price')
    plt.title('Market Price Evolution')
    plt.legend()

    if saveFig:
        plt.savefig('market_price.pdf')
    elif not profiler_view:
        plt.show()

def plot_ratio_buyers_sellers(ratio, profiler_view=False, saveFig=False):
    if profiler_view:
        plt.subplot(3, 1, 2)
    plt.plot(ratio, label='Buyers/Sellers', color='green')
    plt.xlabel('Time Steps')
    plt.ylabel('Ratio')
    plt.title('Buyers/Sellers Ratio')
    plt.legend()
    plt.tight_layout()

    if saveFig:
        plt.savefig('ratio_buyers_sellers.pdf')
    elif not profiler_view:
        plt.show()

def plot_weighted_volumes(weighted_volumes, profiler_view=False, saveFig=False):
    if profiler_view:
        plt.subplot(3, 1, 3)
    
    plt.plot(weighted_volumes)
    plt.ylim(0, 1)
    plt.xlabel('Time Steps')
    plt.ylabel('Volumes Proportion (Buy)')
    
    if saveFig:
        plt.savefig('weighted_volumes.pdf')
    else: # In this case since this is the last plot we show it even if profileview is True
        plt.show()

# def plot_avalanches(all_times, all_sizes):
#     plt.figure(figsize=(15, 6))
#     log_bins = np.logspace(np.log10(min(all_times)), np.log10(max(all_times)), 50)
#     print(all_times)
#     plt.hist(all_times, bins=log_bins, log=True)
#     plt.xlabel('Avalanche Duration')
#     plt.ylabel('Frequency')
#     plt.title('Distribution of Avalanche Durations')
#     plt.xscale('log')
#     plt.show()

#     plt.figure(figsize=(15, 6))
#     log_bins = np.logspace(np.log10(min(all_sizes)), np.log10(max(all_sizes)), 50)
#     plt.hist(all_sizes, bins=log_bins, log=True)
#     plt.xlabel('Price Difference')
#     plt.ylabel('Frequency')
#     plt.title('Avalanches Sizes')
#     plt.xscale('log')
#     plt.show()

def plot_original_vs_filtered_log_returns_pdf(log_returns, filtered_log_returns, bins=50, fit_gaussian_filtered = False):
    original_bin_centers, original_pdf = compute_pdf(log_returns, bins=bins)
    filtered_bin_centers, filtered_pdf = compute_pdf(filtered_log_returns, bins=bins)

    if fit_gaussian_filtered:
        # Fit a Gaussian to the filtered log returns
        mu_filtered, std_filtered = norm.fit(filtered_log_returns)
        gaussian_x = np.linspace(min(log_returns), max(log_returns), 500)
        filtered_gaussian = norm.pdf(gaussian_x, mu_filtered, std_filtered)
        plt.plot(gaussian_x, filtered_gaussian, '--', label="Gaussian Fit (Filtered)", color="green", alpha=0.8)

    # Plot original log returns PDF
    plt.plot(original_bin_centers, original_pdf, '^', label="Original Log Returns", alpha=0.7)

    # Plot filtered log returns PDF
    plt.plot(filtered_bin_centers, filtered_pdf, 'o', label="Filtered Log Returns", alpha=0.7)

    plt.yscale("log")  # Logarithmic scale for better visibility of tails
    plt.xlabel("Log Returns")
    plt.ylabel("Probability Density")
    plt.title("PDF of Logarithmic Returns Before and After Filtering")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()

def plot_avalanches_on_log_returns(log_returns, residual_signal, filtered_log_returns, labeled_array, num_features):
    plt.figure(figsize=(12, 6))
    plt.plot(log_returns, label="Original Log Returns", alpha=0.6)
    plt.plot(residual_signal, label="Residual Signal", alpha=0.8)
    plt.plot(filtered_log_returns, label="Filtered Signal", alpha=0.8)
    for i in range(1, num_features + 1):
        indices = np.where(labeled_array == i)[0]
        plt.axvspan(indices[0], indices[-1], color='red', alpha=0.3, label="Avalanche" if i == 1 else None)
    plt.legend()
    plt.title("Avalanches in Log Returns (Residual Signal)")
    plt.xlabel("Time Steps")
    plt.ylabel("Log Returns")
    plt.show()

def plot_avalanche_sizes(avalanche_sizes):
    """
    Fits the provided avalanche_sizes to a power-law model and plots
    the empirical distribution (PDF) as points (scatter) and the fitted 
    power-law as a solid line.
    Returns (alpha, xmin) and also prints them.
    """
    # Perform the fit
    fit = powerlaw.Fit(avalanche_sizes)
    alpha = fit.alpha
    xmin = fit.xmin
    
    # Plot PDF of empirical data as points (no lines)
    fig_pdf = fit.plot_pdf(
        marker='o',      # Use circle markers
        linestyle='none',# No line connecting points
        color='b',
        label='Empirical Data'
    )
    
    # Plot the power-law fit as a solid line
    fit.power_law.plot_pdf(
        ax=fig_pdf,
        color='r',
        linestyle='-',    # Solid line
        linewidth=2,
        label='Power-Law Fit'
    )

    plt.title("Avalanche Size Distribution (Power-Law Fit)")
    plt.xlabel("Avalanche Size")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.show()

    print(f"Power-law alpha: {alpha}, xmin: {xmin}")
    return alpha, xmin


