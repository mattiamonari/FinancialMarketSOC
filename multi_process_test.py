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
NUM_SIMULATIONS = 8  

# Number of parallel processes (often set to CPU threads; 
# experiment with 8 vs 16 if you have an 8-core/16-thread CPU)
NUM_PROCESSES = 8 



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

    # 7. Wavelet filtering and avalanche extraction
    C_optimal, filtered_log_returns, residual_signal = tune_threshold(log_returns)
    avalanche_sizes, avalanche_durations = extract_avalanches(residual_signal, avalanche_threshold=0.01)

    return {
        'sim_id': sim_id,
        'avalanche_sizes': avalanche_sizes,
        'avalanche_durations': avalanche_durations,
        'C_optimal': C_optimal,
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
            results.append(result)

    # Collect avalanche sizes from all simulations
    all_avalanche_sizes = []
    for res in results:
        all_avalanche_sizes.extend(res['avalanche_sizes'])

    # Plot the aggregated avalanche sizes, get alpha & xmin
    alpha, xmin = plot_avalanche_sizes(all_avalanche_sizes)

    # Print them explicitly here as well
    print(f"Aggregated Avalanche Sizes -> Power-law fit: alpha = {alpha}, xmin = {xmin}")

if __name__ == "__main__":
    main()
