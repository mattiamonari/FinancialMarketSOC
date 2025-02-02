import multiprocessing as mp
import numpy as np
import networkx as nx
import pywt
import sys
from scipy.ndimage import label
from tqdm import tqdm
from fit import *
from wavelet import *
from plots import *
from market_statistics import *
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# GLOBAL PARAMETERS
# ------------------------------
NUM_NODES = 1000
NUM_HEDGE_FUNDS = 10
RANDOM_TRADER_RATIO = 0.25
ALPHA = 0.7
BETA = 0.3
GAMMA = 1
ETA = 0.15
AVALANCHE_THRESHOLD = 0.01
C = 3
HF_TRADE_LOWER = 1
HF_TRADE_UPPER = 2
TR_TRADE_LOWER = 0.2
TR_TRADE_UPPER = 0.6

# Increase TIME_STEPS to gather more data, but note longer runs = longer time
TIME_STEPS = 1000

saveFig = False

def initialize_network():
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
            G.nodes[node]['trade_size'] = np.random.uniform(HF_TRADE_LOWER, HF_TRADE_UPPER)
        elif node in random_trader_indices:
            G.nodes[node]['type'] = 'random_trader'
            G.nodes[node]['trade_size'] = np.random.uniform(TR_TRADE_LOWER, TR_TRADE_UPPER)
        else:
            G.nodes[node]['type'] = 'trader'
            G.nodes[node]['profit_threshold'] = np.random.normal(0.3, 0.1)
            G.nodes[node]['trade_size'] = np.random.uniform(0.2, 0.6)

        G.nodes[node]['last_update_time'] = 0
        G.nodes[node]['position'] = 'buy' if np.random.random() < 0.5 else 'sell'

    return G

G = initialize_network()


def update_positions(t):
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


def update_price(prices, num_buyers, num_sellers):

        price = prices[-1]
        
        buyers = sum(1 for node in G.nodes if G.nodes[node]['position'] == 'buy')
        sellers = sum(1 for node in G.nodes if G.nodes[node]['position'] == 'sell')
   
        num_buyers.append(buyers)
        num_sellers.append(sellers)

        buy_volume = sum(G.nodes[node]['trade_size'] 
                         for node in G.nodes if G.nodes[node]['position'] == 'buy')
        sell_volume = sum(G.nodes[node]['trade_size'] 
                          for node in G.nodes if G.nodes[node]['position'] == 'sell')
        volume_difference = abs(buy_volume - sell_volume)

        # Avoid dividing by zero if volume_difference = 0
        if volume_difference == 0:
            volume_difference = 1

        price += ETA * (buy_volume - sell_volume) * np.random.exponential(1/volume_difference)
        prices.append(price)

        return prices, num_buyers, num_sellers


# ------------------------------
# SINGLE SIMULATION FUNCTION
# ------------------------------
def run_single_simulation(saveFig, sim_id):
    """
    Runs a single market simulation for TIME_STEPS and returns 
    avalanche sizes & durations, along with other simulation data.
    """
    # Optional: set a random seed specific to sim_id for reproducibility
    np.random.seed(sim_id)

    # 3. Initialize market price and containers
    price = 100
    prices = [price]
    num_buyers = []
    num_sellers = []

    # 4. Define update functions

    # 5. Run the simulation
    for t in tqdm(range(1, TIME_STEPS)):
        update_positions(t)
        prices, num_buyers, num_sellers = update_price(prices, num_buyers, num_sellers)

    # Computing the moving average of the market price
    moving_avg = pd.Series(prices).rolling(window=200).mean().to_numpy()
    print("Market volatility: ", np.std(prices))

    returns = calculate_price_returns(prices)
    log_returns = calculate_log_returns(prices)

    # 6. Clean log_returns
    log_returns = np.array(log_returns)
    mask = np.isfinite(log_returns)
    log_returns = log_returns[mask]

    plot_returns(returns, saveFig=saveFig)
    plot_returns(returns**2, saveFig=saveFig, squared=True)
    plot_returns(log_returns, saveFig=saveFig, squared=True, log_returns=True)

    returns_autocorrelation(returns, saveFig=saveFig)
    returns_autocorrelation(returns**2, saveFig=saveFig, squared=True)


    # # 7. Wavelet filtering and avalanche extraction
    # C_optimal, filtered_log_returns, residual_signal = tune_threshold(log_returns)
    # avalanche_sizes, avalanche_durations = extract_avalanches(residual_signal, avalanche_threshold=0.01)

    wavelet = 'db1'	
    level = 4
    if np.any(log_returns):
        coeffs = pywt.wavedec(log_returns, wavelet=wavelet, level=level)
        filtered_coeffs = filter_wavelet_coefficients_paper(coeffs, C)
        filtered_log_returns = pywt.waverec(filtered_coeffs, wavelet='db1', mode='periodization')
        filtered_log_returns = filtered_log_returns[:len(log_returns)]
        residual_signal = log_returns - filtered_log_returns
    else:
        return -1
    avalanche_sizes, avalanche_durations, avalanche_intertimes = extract_avalanches(residual_signal, avalanche_threshold=AVALANCHE_THRESHOLD)
    labeled_array, num_features = label(np.abs(residual_signal) > AVALANCHE_THRESHOLD)

    plot_avalanches_on_log_returns(log_returns, residual_signal, filtered_log_returns, labeled_array, num_features, saveFig=saveFig)
    plot_original_vs_filtered_log_returns_pdf(log_returns, filtered_log_returns, bins=50, fit_gaussian_filtered = True, saveFig=saveFig)

    plot_market_price(prices, moving_avg, profiler_view=False, saveFig=saveFig, num_features=num_features, labeled_array=labeled_array)
    ratio = [num_buyers[i] / (num_sellers[i] + num_buyers[i]) for i in range(len(num_buyers))]
    plot_ratio_buyers_sellers(ratio, profiler_view=False, saveFig=saveFig, num_features=num_features, labeled_array=labeled_array)
    # plot_weighted_volumes(weighted_volumes, profiler_view=True, saveFig=False)

    return {
        'sim_id': sim_id,
        'avalanche_sizes': avalanche_sizes,
        'avalanche_durations': avalanche_durations,
        'avalanche_intertimes': avalanche_intertimes,
        'C': C,
    }

def wrapper(args):
    return run_single_simulation(*args)


# ------------------------------
# PARALLEL EXECUTION MAIN
# ------------------------------
def main():

    global G
    global saveFig

    # Read if use saved data from command line
    # Read number of simulations and processes
    print(len(sys.argv))
    print(sys.argv)
    if len(sys.argv) == 2:
        read_data = bool(int(sys.argv[1]))
    if len(sys.argv) == 5:
        print("Using command line arguments")
        read_data = bool(int(sys.argv[1]))
        saveFig = bool(int(sys.argv[2]))
        NUM_SIMULATIONS = int(sys.argv[3])
        NUM_PROCESSES = int(sys.argv[4])
    else:
        read_data = False
        saveFig = False
        NUM_SIMULATIONS = 1
        NUM_PROCESSES = 1


    # We'll collect results in a list. We'll set chunk_size=1 so that 
    # tqdm can update immediately after each simulation completes.
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        it = pool.imap_unordered(wrapper, [(saveFig, sim_id) for sim_id in range(NUM_SIMULATIONS)], chunksize=1)
        results = []
        for result in tqdm(it, total=NUM_SIMULATIONS, desc="Running simulations in parallel"):
            if result != -1:
                results.append(result)

    # Collect avalanche sizes from all simulations
    all_avalanche_sizes = []
    all_avalanche_durations = []
    all_avalanche_intertimes = []
    if read_data:
        all_avalanche_sizes = np.genfromtxt('csv/example_avalanche_sizes.csv', delimiter=',')
        all_avalanche_durations = np.genfromtxt('csv/example_avalanche_durations.csv', delimiter=',')
        all_avalanche_intertimes = np.genfromtxt('csv/example_avalanche_intertimes.csv', delimiter=',')
    else :
        for res in results:
            all_avalanche_sizes.extend(res['avalanche_sizes'])
            all_avalanche_durations.extend(res['avalanche_durations'])
            all_avalanche_intertimes.extend(res['avalanche_intertimes'])

    all_avalanche_sizes = np.array(all_avalanche_sizes)
    all_avalanche_durations = np.array(all_avalanche_durations)
    all_avalanche_intertimes = np.array(all_avalanche_intertimes)

    if not read_data:
        np.savetxt('csv/avalanche_sizes.csv', all_avalanche_sizes, delimiter=',')
        np.savetxt('csv/avalanche_durations.csv', all_avalanche_durations, delimiter=',')
        np.savetxt('csv/avalanche_intertimes.csv', all_avalanche_intertimes, delimiter=',')   
    
    # Optional: filter out zero/negative values if that makes no physical sense:
    all_avalanche_sizes = all_avalanche_sizes[all_avalanche_sizes > 0]
    all_avalanche_durations = all_avalanche_durations[all_avalanche_durations > 0]
    all_avalanche_intertimes = all_avalanche_intertimes[all_avalanche_intertimes > 0]

    fit_curve_power_law(all_avalanche_sizes, lower_cutoff=0.7, upper_cutoff=30, 
                          xlabel="Avalanche Sizes $|P_{t+1}-P_t|$",
                          title="Log-Binned Avalanche Sizes",
                          name="avalanche_sizes", saveFig=saveFig)
    

    fit_curve_exponential(all_avalanche_durations, lower_cutoff=12, upper_cutoff=1e6, 
                        xlabel="Avalanche Durations (timesteps)",
                        title="Log-Binned Avalanche Durations",
                        name="avalanche_durations", saveFig=saveFig)

    fit_curve_power_law(all_avalanche_intertimes, lower_cutoff=0.08, upper_cutoff=6000.0, 
                        xlabel="Time Between Avalanches (timesteps)",
                        title="Log-Binned Avalanche In-Between Times",
                        name="avalanche_intertimes", saveFig=saveFig)


if __name__ == "__main__":
    main()
