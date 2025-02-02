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
from plots import *
import pandas as pd

NUM_NODES = 1000 # Total number of traders (including hedge funds)
NUM_HEDGE_FUNDS = 10  # Number of hedge funds (high-degree nodes)
RANDOM_TRADER_RATIO = 0.25  # Ratio of traders who act randomly
ALPHA = 0.7  # Weight for trade size influence
BETA = 0.3  # Weight for degree influence
GAMMA = 1  # Sensitivity for profit acceptance
ETA = 0.2  # Scaling factor for price changes
TIME_STEPS = 500  # Number of time steps for the simulation

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
price = 100
prices = [price]

# Keep track of returns
returns = []
log_returns = []  

# Keep track of number of buyers and sellers
num_buyers = []
num_sellers = []

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
    global num_buyers
    global num_sellers
    buy_volume = sum(G.nodes[node]['trade_size'] for node in G.nodes if G.nodes[node]['position'] == 'buy')
    sell_volume = sum(G.nodes[node]['trade_size'] for node in G.nodes if G.nodes[node]['position'] == 'sell')
    volume_difference = abs(buy_volume - sell_volume)
    price += ETA * (buy_volume - sell_volume) * np.random.exponential(1/volume_difference)
    prices.append(price)
    buyers = sum(1 for node in G.nodes if G.nodes[node]['position'] == 'buy')
    sellers = sum(1 for node in G.nodes if G.nodes[node]['position'] == 'sell')

    num_buyers.append(buyers)
    num_sellers.append(sellers)

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
moving_avg = pd.Series(prices).rolling(window=200).mean().to_numpy()
print(moving_avg)
plot_market_price(prices, moving_avg, profiler_view=False, saveFig=False)

# Truncate the returns in the range (-0.1, 0.1) for better visualization
returns = [r for r in returns if -0.25 <= r <= 0.25]
plot_returns_distribution(returns, fitted_dist=True, saveFig=False, log_returns=False)


log_returns = [lr for lr in log_returns if np.isfinite(lr) and -0.5 <= lr <= 0.5]
plot_returns_distribution(log_returns, fitted_dist=True, saveFig=False, log_returns=True)

wavelet = 'db1'	
level = 4
coeffs = pywt.wavedec(log_returns, wavelet=wavelet, level=level)
filtered_coeffs = filter_wavelet_coefficients_paper(coeffs, C=3.0)
filtered_log_returns = pywt.waverec(filtered_coeffs, wavelet='db1', mode='periodization')
filtered_log_returns = filtered_log_returns[:len(log_returns)]
residual_signal = log_returns - filtered_log_returns


# Print the optimal C and plot the results
# print(f"Optimal C: {C_optimal}")

# Detect avalanches based on the residual signal
avalanche_threshold = 0.01  # Set threshold for high activity
labeled_array, num_features = label(np.abs(residual_signal) > avalanche_threshold)

avalanche_sizes, avalanche_durations = extract_avalanches(residual_signal)
# # Analyze avalanche size distribution
# fit = powerlaw.Fit(avalanche_sizes)
# print(f"Power-law alpha: {fit.alpha}, xmin: {fit.xmin}")

# log_bin_avalanche_sizes, log_bin_centers, total_hits = histogram_log_bins(avalanche_sizes, num_of_bins=50, min_hits=1)
# # Visualize avalanche size distribution with power-law fit
# fit.plot_pdf(color='b', linewidth=2)
# fit.power_law.plot_pdf(color='r', linestyle='--')
# plt.title("Avalanche Size Distribution (Power-law Fit)")
# plt.xlabel("Avalanche Size")
# plt.ylabel("Probability Density")
# plt.scatter(log_bin_centers, log_bin_avalanche_sizes / total_hits, color='g', label='Logarithmic Binning')
# plt.show()


# Visualize the original signal, filtered signal, and avalanches and residuals
plot_avalanches_on_log_returns(log_returns, residual_signal, filtered_log_returns, labeled_array, num_features)
plot_original_vs_filtered_log_returns_pdf(log_returns, filtered_log_returns, bins=50, fit_gaussian_filtered = True, saveFig=False)


log_returns = np.array(log_returns)

plot_market_price(prices, moving_avg, profiler_view=True, saveFig=False, num_features=num_features, labeled_array=labeled_array)
ratio = [num_buyers[i] / (num_sellers[i] + num_buyers[i]) for i in range(len(num_buyers))]
plot_ratio_buyers_sellers(ratio, profiler_view=True, saveFig=False, num_features=num_features, labeled_array=labeled_array)
# plot_weighted_volumes(weighted_volumes, profiler_view=True, saveFig=False)
plt.show()

returns_autocorrelation(np.array(returns), saveFig=False)
returns_autocorrelation(np.array(returns)** 2, saveFig=False, squared=True)

