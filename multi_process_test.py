import multiprocessing as mp
import numpy as np
import networkx as nx
import pywt
import powerlaw
from scipy.stats import norm, kurtosis, skew
from scipy.ndimage import label
from tqdm import tqdm
import matplotlib.pyplot as plt
from wavelet import *
from plots import *


# ------------------------------
# GLOBAL PARAMETERS
# ------------------------------
NUM_NODES = 1000
NUM_HEDGE_FUNDS = 10
RANDOM_TRADER_RATIO = 0.25
ALPHA = 0.7
BETA = 0.3
GAMMA = 1
ETA = 0.01

# Increase TIME_STEPS to gather more data, but note longer runs = longer time
TIME_STEPS = 10000

# Number of simulations to run in parallel
NUM_SIMULATIONS = 64

# Number of parallel processes (often set to CPU threads; 
# experiment with 8 vs 16 if you have an 8-core/16-thread CPU)
NUM_PROCESSES = 8 




def histogram_log_bins(x, x_min=None, x_max=None, num_of_bins=20, min_hits=1):
    """
    Generate histogram with logarithmically spaced bins.
    """
    if not x_min:
        x_min = np.min(x)
    if not x_max:
        x_max = np.max(x)

    # This is the factor that each subsequent bin is larger than the next.
    growth_factor = (x_max / x_min) ** (1 / (num_of_bins + 1))
    # Generates logarithmically spaced points from x_min to x_max.
    bin_edges = np.logspace(np.log10(x_min), np.log10(x_max), num=num_of_bins + 1)
    # We don't need the second argument (which are again the bin edges).
    # It's conventional to denote arguments you don't intend to use with _.
    bin_counts, _ = np.histogram(x, bins=bin_edges)
    total_hits = np.sum(bin_counts)
    bin_counts = bin_counts.astype(float)

    # Rescale bin counts by their relative sizes.
    significant_bins = []
    for bin_index in range(np.size(bin_counts)):
        if bin_counts[bin_index] >= min_hits:
            significant_bins.append(bin_index)

        bin_counts[bin_index] = bin_counts[bin_index] / (growth_factor ** bin_index)

    # Is there a better way to get the center of a bin on logarithmic axis? There probably is, please figure it out.
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # You can optionally rescale the counts by total_hits if you want to get a density.
    return bin_counts[significant_bins], bin_centers[significant_bins], total_hits


# ------------------------------
# SINGLE SIMULATION FUNCTION
# ------------------------------
def run_single_simulation(sim_id):
    """
    Runs a single market simulation for TIME_STEPS and returns 
    avalanche sizes & durations, along with other simulation data.
    """
    # Optional: set a random seed specific to sim_id for reproducibility
    np.random.seed(sim_id)

    # 1. Construct scale-free network
    G = nx.barabasi_albert_graph(NUM_NODES, m=5)

    # 2. Assign node attributes
    num_random_traders = int(NUM_NODES * RANDOM_TRADER_RATIO)
    random_trader_indices = np.random.choice(
        range(NUM_HEDGE_FUNDS, NUM_NODES),
        size=num_random_traders, replace=False
    )
    for node in G.nodes:
        if node < NUM_HEDGE_FUNDS:
            G.nodes[node]['type'] = 'hedge_fund'
            G.nodes[node]['profit_threshold'] = np.random.normal(0.3, 0.1)
            G.nodes[node]['trade_size'] = np.random.uniform(1, 2)
        elif node in random_trader_indices:
            G.nodes[node]['type'] = 'random_trader'
            G.nodes[node]['trade_size'] = np.random.uniform(0.2, 0.6)
        else:
            G.nodes[node]['type'] = 'trader'
            G.nodes[node]['profit_threshold'] = np.random.normal(0.3, 0.1)
            G.nodes[node]['trade_size'] = np.random.uniform(0.2, 0.6)

        G.nodes[node]['last_update_time'] = 0
        G.nodes[node]['position'] = 'buy' if np.random.random() < 0.5 else 'sell'

    # 3. Initialize market price and containers
    price = 0
    prices = [price]
    returns = []
    log_returns = []

    # 4. Define update functions
    def update_positions(t):
        nonlocal price
        for node in G.nodes:
            neighbors = list(G.neighbors(node))
            if G.nodes[node]['type'] == 'hedge_fund':
                profit = np.random.uniform(0,1)
                # Switch position if profit threshold is exceeded
                if np.random.rand() < 1 / (1 + np.exp(- (GAMMA * (profit - G.nodes[node]['profit_threshold'])))):
                    G.nodes[node]['position'] = (
                        'buy' if G.nodes[node]['position'] == 'sell' else 'sell'
                    )
                    G.nodes[node]['last_update_time'] = t

            elif G.nodes[node]['type'] == 'trader':
                for neighbor in neighbors:
                    influence = ALPHA * G.nodes[neighbor]['trade_size'] / 10 + BETA * len(neighbors)
                    if np.random.rand() < influence and G.nodes[node]['last_update_time'] < t:
                        G.nodes[node]['position'] = G.nodes[neighbor]['position']
                        G.nodes[node]['last_update_time'] = t

            elif G.nodes[node]['type'] == 'random_trader':
                G.nodes[node]['position'] = 'buy' if np.random.rand() < 0.5 else 'sell'
                G.nodes[node]['last_update_time'] = t

    def update_price(t):
        nonlocal price
        buy_volume = sum(G.nodes[node]['trade_size'] 
                         for node in G.nodes if G.nodes[node]['position'] == 'buy')
        sell_volume = sum(G.nodes[node]['trade_size'] 
                          for node in G.nodes if G.nodes[node]['position'] == 'sell')
        volume_difference = abs(buy_volume - sell_volume)

        # Avoid dividing by zero if volume_difference = 0
        if volume_difference == 0:
            return

        price += ETA * (buy_volume - sell_volume) * np.random.exponential(1/volume_difference)
        prices.append(price)
        if t > 1 and prices[-2] != 0:
            ret = (prices[-1] - prices[-2]) / prices[-2]
            returns.append(ret)
            # Log returns
            if prices[-1] > 0 and prices[-2] > 0:
                log_returns.append(np.log(prices[-1] / prices[-2]))
            else:
                log_returns.append(np.nan)

    # 5. Run the simulation
    for t in tqdm(range(1, TIME_STEPS)):
        update_positions(t)
        update_price(t)

    # 6. Clean log_returns
    log_returns = np.array(log_returns)
    mask = np.isfinite(log_returns)
    log_returns = log_returns[mask]

    # # 7. Wavelet filtering and avalanche extraction
    # C_optimal, filtered_log_returns, residual_signal = tune_threshold(log_returns)
    # avalanche_sizes, avalanche_durations = extract_avalanches(residual_signal, avalanche_threshold=0.01)

    wavelet = 'db1'	
    level = 4
    C = 3
    if np.any(log_returns):
        coeffs = pywt.wavedec(log_returns, wavelet=wavelet, level=level)
        filtered_coeffs = filter_wavelet_coefficients_paper(coeffs, C)
        filtered_log_returns = pywt.waverec(filtered_coeffs, wavelet='db1', mode='periodization')
        filtered_log_returns = filtered_log_returns[:len(log_returns)]
        residual_signal = log_returns - filtered_log_returns
    else:
        return -1
    avalanche_sizes, avalanche_durations, avalanche_intertimes = extract_avalanches(residual_signal, avalanche_threshold=0.01)

    return {
        'sim_id': sim_id,
        'avalanche_sizes': avalanche_sizes,
        'avalanche_durations': avalanche_durations,
        'avalanche_intertimes': avalanche_intertimes,
        'C': C,
    }

# ------------------------------
# PARALLEL EXECUTION MAIN
# ------------------------------
def main():
    # We'll collect results in a list. We'll set chunk_size=1 so that 
    # tqdm can update immediately after each simulation completes.
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        it = pool.imap_unordered(run_single_simulation, range(NUM_SIMULATIONS), chunksize=1)
        results = []
        for result in tqdm(it, total=NUM_SIMULATIONS, desc="Running simulations in parallel"):
            if result != -1:
                results.append(result)

    # Collect avalanche sizes from all simulations
    all_avalanche_sizes = []
    all_avalanche_durations = []
    all_avalanche_intertimes = []
    for res in results:
        all_avalanche_sizes.extend(res['avalanche_sizes'])
        all_avalanche_durations.extend(res['avalanche_durations'])
        all_avalanche_intertimes.extend(res['avalanche_intertimes'])

    # save data as csv file
    np.savetxt('avalanche_sizes.csv', all_avalanche_sizes, delimiter=',')
    np.savetxt('avalanche_durations.csv', all_avalanche_durations, delimiter=',')
    np.savetxt('avalanche_intertimes.csv', all_avalanche_intertimes, delimiter=',')

    
    all_avalanche_sizes_log_bins = histogram_log_bins(all_avalanche_sizes, num_of_bins=50, min_hits=1)

    plt.scatter(all_avalanche_sizes_log_bins[1], all_avalanche_sizes_log_bins[0] / all_avalanche_sizes_log_bins[2])
    plt.xlabel('Avalanche Size')
    plt.ylabel('Frequency')
    plt.title('Avalanche Size Distribution')   
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    all_avalanche_durations_log_bins = histogram_log_bins(all_avalanche_durations, num_of_bins=50, min_hits=1)

    plt.scatter(all_avalanche_durations_log_bins[1], all_avalanche_durations_log_bins[0] / all_avalanche_durations_log_bins[2])
    plt.xlabel('Avalanche duration')
    plt.ylabel('Frequency')
    plt.title('Avalanche Duration Distribution')   
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    all_avalanche_intertimes_log_bins = histogram_log_bins(all_avalanche_intertimes, num_of_bins=50, min_hits=1)

    plt.scatter(all_avalanche_intertimes_log_bins[1], all_avalanche_intertimes_log_bins[0] / all_avalanche_intertimes_log_bins[2])
    plt.xlabel('Avalanche inbetween times')
    plt.ylabel('Frequency')
    plt.title('Avalanche Inbetween Times Distribution')   
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


    
    # # Plot the aggregated avalanche sizes, get alpha & xmin
    # alpha, xmin = plot_avalanche_sizes(all_avalanche_sizes)

    # Print them explicitly here as well
    # print(f"Aggregated Avalanche Sizes -> Power-law fit: alpha = {alpha}, xmin = {xmin}")

if __name__ == "__main__":
    main()
