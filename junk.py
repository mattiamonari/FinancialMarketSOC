import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
from scipy.stats import kurtosis, skew
from scipy.ndimage import label
import pywt
import powerlaw
from stylised_facts import *

NUM_NODES = 1000 # Total number of traders (including hedge funds)
NUM_HEDGE_FUNDS = 10  # Number of hedge funds (high-degree nodes)
RANDOM_TRADER_RATIO = 0.25  # Ratio of traders who act randomly
ALPHA = 0.7  # Weight for trade size influence
BETA = 0.3  # Weight for degree influence
GAMMA = 1  # Sensitivity for profit acceptance
ETA = 0.01  # Scaling factor for price changes
TIME_STEPS = 1000  # Number of time steps for the simulation

# Initialize a scale-free network using Barabási-Albert model
G = nx.barabasi_albert_graph(NUM_NODES, m=5)

# Assign node attributes (traders, hedge funds, and random traders)
num_random_traders = int(NUM_NODES * RANDOM_TRADER_RATIO)
random_trader_indices = np.random.choice(range(NUM_HEDGE_FUNDS, NUM_NODES), size=num_random_traders, replace=False)

for node in G.nodes:
    if node < NUM_HEDGE_FUNDS:
        G.nodes[node]['type'] = 'hedge_fund'
        G.nodes[node]['profit_threshold'] = np.random.normal(0.3, 0.1)  # Example profit threshold
        G.nodes[node]['trade_size'] = np.random.uniform(1, 2)  # Random trade size
    elif node in random_trader_indices:
        G.nodes[node]['type'] = 'random_trader'
        G.nodes[node]['trade_size'] = np.random.uniform(0.2, 0.6)  # Random trade size
    else:
        G.nodes[node]['type'] = 'trader'
        G.nodes[node]['profit_threshold'] = np.random.normal(0.3, 0.1)  # Example profit threshold (not used at the moment)
        G.nodes[node]['trade_size'] = np.random.uniform(0.2, 0.6)  # Random trade size

    G.nodes[node]['last_update_time'] = 0
    G.nodes[node]['position'] = 'buy' if np.random.random() < 0.5 else 'sell'

# Initialize market price
price = 0
prices = [price]

# Keep track of returns
returns = []
log_returns = []  

# Function to update positions
def update_positions(t):
    global price
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        
        if G.nodes[node]['type'] == 'hedge_fund':
            # Hedge funds evaluate profit and decide to change position
            profit = np.random.uniform(0, 1)  # Random profit for demonstration
            if np.random.rand() < 1 / (1 + np.exp(-GAMMA * (profit - G.nodes[node]['profit_threshold']))):
                G.nodes[node]['position'] = 'buy' if G.nodes[node]['position'] == 'sell' else 'sell'
                G.nodes[node]['last_update_time'] = t

        elif G.nodes[node]['type'] == 'trader':
            # Traders are influenced by neighbors
            for neighbor in neighbors:
                influence = ALPHA * G.nodes[neighbor]['trade_size'] / 10 + BETA * len(neighbors)
                if np.random.rand() < influence and G.nodes[node]['last_update_time'] < t:
                    G.nodes[node]['position'] = G.nodes[neighbor]['position']
                    G.nodes[node]['last_update_time'] = t

        elif G.nodes[node]['type'] == 'random_trader':
            # Random traders change positions randomly
            if np.random.rand() < 0.5:
                G.nodes[node]['position'] = 'buy'
            else:
                G.nodes[node]['position'] = 'sell'
            G.nodes[node]['last_update_time'] = t

# Function to update market price
def update_price(t):
    global price
    buy_volume = sum(G.nodes[node]['trade_size'] for node in G.nodes if G.nodes[node]['position'] == 'buy')
    sell_volume = sum(G.nodes[node]['trade_size'] for node in G.nodes if G.nodes[node]['position'] == 'sell')
    volume_difference = abs(buy_volume - sell_volume)
    price += ETA * (buy_volume - sell_volume) * np.random.exponential(1/volume_difference)
    prices.append(price)

    # Calculate returns
    if t > 1:
        if prices[-2] != 0:  # Avoid division by zero
            returns.append((prices[-1] - prices[-2]) / prices[-2])
            log_returns.append(np.log(prices[-1] / prices[-2]))

# Run the simulation
for t in tqdm(range(1, TIME_STEPS)):
    update_positions(t)
    update_price(t)

# Compute the moving average of the market price
moving_avg = np.convolve(prices, np.ones(25) / 25, mode='valid')

print("Market volatility: ", np.std(prices))

# Plot the price evolution
plt.plot(prices)
plt.plot(moving_avg)
plt.xlabel('Time Steps')
plt.ylabel('Market Price')
plt.title('Market Price Evolution')
plt.show()

# Plot the returns histogram and fit a normal distribution and also the same for log returns
plt.figure(figsize=(10,5))

# Plot histogram of returns
plt.subplot(1,2,1)
# Truncate the returns in the range (-0.1, 0.1) for better visualization
returns = [r for r in returns if -0.1 <= r <= 0.1]
plt.hist(returns, bins=30,density=True, alpha=0.6, color='g')

# Fit a normal distribution to the returns
mu, std = norm.fit(returns)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title("Returns distribution: mu = %.2f,  std = %.2f" % (mu, std))

# Plot histogram of log returns
plt.subplot(1, 2, 2)
# Remove non-finite values from log_returns
log_returns = [lr for lr in log_returns if np.isfinite(lr) and -0.5 <= lr <= 0.5]
plt.hist(log_returns, bins=50, range=(-.5,.5), density=True, alpha=0.6, color='g')

# Fit a normal distribution to the log returns
mu, std = norm.fit(log_returns)

# Plot the fitted normal distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 1000)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.legend(['Normal Distribution Fit', 'Log Returns'])
plt.title("Log Returns distribution: mu = %.2f,  std = %.2f" % (mu, std))
plt.show()


# Function to filter wavelet coefficients based on the paper's method
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

# Function to dynamically tune the threshold C
def tune_threshold(log_returns, wavelet='db1', level=4, target_kurtosis=3, target_skew=0, tolerance=0.1):
    """
    Dynamically tune the threshold parameter C for wavelet filtering based on kurtosis and skewness.
    """
    coeffs = pywt.wavedec(log_returns, wavelet=wavelet, level=level)
    C = 1  # Start with an initial threshold factor
    max_iter = 5  # Limit the number of iterations
    results = []
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
        C += 1
    #print(coeffs)
        results.append([C, filtered_signal, residual_signal])
    return results

# Perform wavelet filtering with dynamically tuned threshold
results = tune_threshold(log_returns)

C = [result[0] for result in results]
filtered_log_returns = [result[1] for result in results]
residual_signal = [result[2] for result in results]


# Print the optimal C and plot the results
# print(f"Optimal C: {C_optimal}")

# Detect avalanches based on the residual signal
avalanche_threshold = 0.01  # Set threshold for high activity
labeled_array, num_features = label(np.abs(residual_signal) > avalanche_threshold)




def extract_avalanches(residual_signal, avalanche_threshold=0.01):
    """
    Extracts avalanches from the residual signal based on a threshold.
    Returns the avalanche sizes and durations.
    """
    labeled_array, num_features = label(np.abs(residual_signal) > avalanche_threshold)
    avalanche_sizes = []
    avalanche_durations = []
    for i in range(1, num_features + 1):
        indices = np.where(labeled_array == i)[0]
        size = np.sum(np.abs(residual_signal[indices]))  # Sum of residuals during the avalanche
        duration = len(indices)  # Number of time steps in the avalanche
        avalanche_sizes.append(size)
        avalanche_durations.append(duration)
    return avalanche_sizes, avalanche_durations




# avalanche_sizes, avalanche_durations = extract_avalanches(residual_signal)
# # Analyze avalanche size distribution
# fit = powerlaw.Fit(avalanche_sizes)
# print(f"Power-law alpha: {fit.alpha}, xmin: {fit.xmin}")

# # Visualize avalanche size distribution with power-law fit
# fit.plot_pdf(color='b', linewidth=2)
# fit.power_law.plot_pdf(color='r', linestyle='--')
# plt.title("Avalanche Size Distribution (Power-law Fit)")
# plt.xlabel("Avalanche Size")
# plt.ylabel("Probability Density")
# #plt.show()

# # Visualize the original signal, filtered signal, and avalanches and residuals
# plt.figure(figsize=(12, 6))
# plt.plot(log_returns, label="Original Log Returns", alpha=0.6)
# plt.plot(residual_signal, label="Residual Signal", alpha=0.8)
# plt.plot(filtered_log_returns, label="Filtered Signal", alpha=0.8)
# for i in range(1, num_features + 1):
#     indices = np.where(labeled_array == i)[0]
#     plt.axvspan(indices[0], indices[-1], color='red', alpha=0.3, label="Avalanche" if i == 1 else None)
# plt.legend()
# plt.title("Avalanches in Log Returns (Residual Signal)")
# plt.xlabel("Time Steps")
# plt.ylabel("Log Returns")
# #plt.show()

# Compute PDF for original and filtered log returns
def compute_pdf(data, bins=50):
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist

mu_original, std_original = norm.fit(log_returns)
gaussian_x = np.linspace(min(log_returns), max(log_returns), 500)
for i in range(len(C)):
    avalanche_sizes, avalanche_durations = extract_avalanches(residual_signal[i])
    filtered_bin_centers, filtered_pdf = compute_pdf(filtered_log_returns[i], bins=100)
    mu_filtered, std_filtered = norm.fit(filtered_log_returns[i])
    filtered_gaussian = norm.pdf(gaussian_x, mu_filtered, std_filtered)
    filtered_bin_centers, filtered_pdf = compute_pdf(filtered_log_returns[i], bins=100)


    plt.plot(filtered_bin_centers, filtered_pdf, 'o', label=f"Filtered Log Returns (C={C[i]})", alpha=0.7)
    plt.plot(gaussian_x, filtered_gaussian, '--', label="Gaussian Fit (Filtered)", color="green", alpha=0.8)
    plt.xlabel("Log Returns")
    plt.ylabel("Probability Density")
    plt.title("PDF of Logarithmic Returns Before and After Filtering")
    plt.legend()
    plt.grid(alpha=0.3)
plt.hist(log_returns, bins=50, range=(-.5,.5), density=True, alpha=0.6, color='g')
plt.yscale("log")  # Logarithmic scale for better visibility of tails
plt.show()


# # Gaussian fit for comparison
# mu_original, std_original = norm.fit(log_returns)
# mu_filtered, std_filtered = norm.fit(filtered_log_returns)

# gaussian_x = np.linspace(min(log_returns), max(log_returns), 500)
# original_gaussian = norm.pdf(gaussian_x, mu_original, std_original)
# filtered_gaussian = norm.pdf(gaussian_x, mu_filtered, std_filtered)

# # Compute PDFs for original and filtered log returns
# bins = 50
# original_bin_centers, original_pdf = compute_pdf(log_returns, bins=bins)
# filtered_bin_centers, filtered_pdf = compute_pdf(filtered_log_returns, bins=bins)

# # Plot the results
# plt.figure(figsize=(10, 6))

# # Plot original log returns PDF
# plt.plot(original_bin_centers, original_pdf, '^', label="Original Log Returns", alpha=0.7)

# # Plot filtered log returns PDF
# plt.plot(filtered_bin_centers, filtered_pdf, 'o', label="Filtered Log Returns", alpha=0.7)

# # Plot Gaussian for comparison
# # plt.plot(gaussian_x, original_gaussian, '-', label="Gaussian Fit (Original)", color="blue", alpha=0.8)
# plt.plot(gaussian_x, filtered_gaussian, '--', label="Gaussian Fit (Filtered)", color="green", alpha=0.8)

# # Add labels and legend
# plt.yscale("log")  # Logarithmic scale for better visibility of tails
# plt.xlabel("Log Returns")
# plt.ylabel("Probability Density")
# plt.title("PDF of Logarithmic Returns Before and After Filtering")
# plt.legend()
# plt.grid(alpha=0.3)



# plt.show()
# log_returns = np.array(log_returns)
# returns_autocorrelation(log_returns, saveFig=False)
# returns_autocorrelation(log_returns**2, saveFig=False)

