import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from market_statistics import * 
from plots import *
from wavelet import *
import multiprocessing

from stylised_facts import *

# Parameters for the simulation
NUM_NODES = 1000  # Total number of traders (including hedge funds)
NUM_HEDGE_FUNDS = 10  # Number of hedge funds (high-degree nodes)
ALPHA = 0.05  # Weight for trade size influence
BETA = 0.3  # Weight for degree influence
GAMMA = 1  # Sensitivity for profit acceptance
ETA = 0.01  # Scaling factor for price changes
EXPONENTIAL_SCALING = 1.12 # Exponential scaling factor for high volume differences
TIME_STEPS = 5000  # Number of time steps for the simulation
N_RUNS = 1 # Number of repetitions of the experiment
HF_TRADE_LOWER = 10
HF_TRADE_UPPER = 50
TR_TRADE_LOWER = 0.2
TR_TRADE_UPPER = 1

# Initialize a scale-free network using Barabási-Albert model
G = nx.barabasi_albert_graph(NUM_NODES, m=NUM_HEDGE_FUNDS)

# Initialize lists to track the number of buyers and sellers
random_lags = np.random.randint(1, 5, size=TIME_STEPS)

def initialize_graph(G):
    # Order nodes by degree and assign attributes. This is done to ensure that hedge funds are high-degree nodes.
    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)  # Sort nodes by degree
    for idx, (node, degree) in enumerate(sorted_nodes):
        if idx < NUM_HEDGE_FUNDS:
            G.nodes[node]['type'] = 'hedge_fund'
            G.nodes[node]['profit_threshold'] = np.random.normal(0.5, 0.1)  # Example profit threshold
            G.nodes[node]['trade_size'] = np.random.uniform(HF_TRADE_LOWER, HF_TRADE_UPPER)  # Random trade size
        else:
            G.nodes[node]['type'] = 'trader'
            G.nodes[node]['profit_threshold'] = np.random.normal(0.3, 0.1)  # Example profit threshold (not used at the moment)
            G.nodes[node]['trade_size'] = np.random.uniform(TR_TRADE_LOWER, TR_TRADE_UPPER)  # Random trade size
        
        G.nodes[node]['last_update_time'] = 0
        G.nodes[node]['position'] = 'buy' if np.random.random() < 0.5 else 'sell'

    return G

def update_positions(G, t):
    global random_lags
    
    #Calculate max degree without hedge funds
    max_degree = max([d for n, d in G.degree if G.nodes[n]['type'] != 'hedge_fund'])

    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        
        if G.nodes[node]['type'] == 'hedge_fund':
            # Hedge funds evaluate profit and decide to change position
            profit = np.random.uniform(0, 1)  # Random profit for demonstration
            if np.random.rand() < 1 / (1 + np.exp(-GAMMA * (profit - G.nodes[node]['profit_threshold']))):
                G.nodes[node]['position'] = 'buy' if G.nodes[node]['position'] == 'sell' else 'sell'
                G.nodes[node]['last_update_time'] = t
                G.nodes[node]['trade_size'] = np.random.uniform(10, 50)  # Random trade size
        
        elif G.nodes[node]['type'] == 'trader':
            influence_sum = 0
            for neighbor in neighbors:
                influence = ALPHA * G.nodes[neighbor]['trade_size'] / 50 + 1/max_degree * G.degree[neighbor]
                influence_sum += influence   
        
            avg_influence = influence_sum / len(neighbors)
            last_update_time = G.nodes[node]['last_update_time']
            if (np.random.rand() < avg_influence and t - last_update_time >= np.random.randint(1, 5)):
                influential_neighbor = max(neighbors, key=lambda n: G.degree[n])
                G.nodes[node]['position'] = G.nodes[influential_neighbor]['position']
                G.nodes[node]['last_update_time'] = t
                G.nodes[node]['trade_size'] = np.random.uniform(0.2, 1)  # Random trade size

def update_price(G, price, num_buyers, num_sellers):
    buy_volume = sum(G.nodes[node]['trade_size'] for node in G.nodes if G.nodes[node]['position'] == 'buy')
    sell_volume = sum(G.nodes[node]['trade_size'] for node in G.nodes if G.nodes[node]['position'] == 'sell')
    # Count the number of buyers and sellers
    buyers = sum(1 for node in G.nodes if G.nodes[node]['position'] == 'buy')
    sellers = sum(1 for node in G.nodes if G.nodes[node]['position'] == 'sell')
   
    num_buyers.append(buyers)
    num_sellers.append(sellers)
   
    volume_difference = abs(buy_volume - sell_volume)
    # Apply exponential scaling for large volume differences
    if volume_difference > (buy_volume + sell_volume) / 2.6:  # Example threshold
        price += ETA * np.sign(buy_volume - sell_volume) * (volume_difference) ** EXPONENTIAL_SCALING
    else:
        price += ETA * (buy_volume - sell_volume)

    return price

def run_simulation():
    price = 1000
    prices = [price]
    num_buyers = []
    num_sellers = []
    buy_volumes = np.zeros(TIME_STEPS)
    sell_volumes = np.zeros(TIME_STEPS)

    G = nx.barabasi_albert_graph(NUM_NODES, m=NUM_HEDGE_FUNDS)
    G = initialize_graph(G)

    # Run the simulation
    for t in range(TIME_STEPS):
        update_positions(G, t)
        prices.append(update_price(G, prices[-1], num_buyers, num_sellers))
        buy_volumes[t], sell_volumes[t] = calculate_volumes(G)

    returns = calculate_price_returns(prices)
    print("The mean of returns is: ", np.mean(returns))

    moving_avg = np.convolve(prices, np.ones(10) / 10, mode='valid')
    print("Market volatility: ", np.std(prices))

    starts, ends, times = detect_avalanche(moving_avg, threshold_start=0.0015, threshold_end=0.001, num_buyers=num_buyers, num_sellers=num_sellers, 
                                           buy_volumes=buy_volumes, sell_volumes=sell_volumes)
    
    plt.plot(prices, label='Market Price', color='blue')
    plt.plot(moving_avg, label='Moving Average', color='red')
        
    for i in range(len(starts)):
        plt.vlines(starts[i], 800, 1200, colors='red', linestyles='dashed', linewidth=0.5)

    for i in range(len(ends)):
        plt.vlines(ends[i], 800, 1200, colors='green', linestyles='dashed', linewidth=0.5)

    if len(ends) != len(starts):
        ends.append(len(prices) -  1)
    
    price_starts = [prices[i] for i in starts]
    price_ends = [prices[i] for i in ends]
    price_diff = [np.abs(price_ends[i] - price_starts[i]) for i in range(len(price_starts))]    
    return price_diff, times

def main():
    weighted_volumes = np.zeros(TIME_STEPS)
    price = 1000
    prices = [price]
    num_buyers = []
    num_sellers = []

    G = nx.barabasi_albert_graph(NUM_NODES, m=NUM_HEDGE_FUNDS)
    G = initialize_graph(G)
    for t in tqdm(range(TIME_STEPS), desc="Time step"):
        update_positions(G, t)
        prices.append(update_price(G, prices[-1], num_buyers, num_sellers))
        weighted_volumes[t] = calculate_volume_imbalance(G)

    # Compute the moving average of the market price
    moving_avg = np.convolve(prices, np.ones(10) / 10, mode='valid')
    print("Market volatility: ", np.std(prices))

    # Added for the 'profile view' in the plot. If in the future we remove it, change the method parameter and remove this.
    plot_market_price(prices, moving_avg, profiler_view=True, saveFig=False)
    ratio = [num_buyers[i] / (num_sellers[i] + num_buyers[i]) for i in range(len(num_buyers))]
    plot_ratio_buyers_sellers(ratio, profiler_view=True, saveFig=False)
    plot_weighted_volumes(weighted_volumes, profiler_view=True, saveFig=False)

    returns = calculate_price_returns(prices)
    plot_returns(returns, saveFig=False)

    plot_returns_vs_time(returns**2, saveFig=False)

    returns_autocorrelation(returns, saveFig=False)
    returns_autocorrelation(returns**2, saveFig=False)

    all_times = []
    all_sizes = []
    print(f"Running {N_RUNS} runs")
    
    run_simulation()

    # TODO: Uncomment this part to run multiple simulations and extract power law 
    #       distributions for the avalanches
    # with multiprocessing.Pool(10) as p:
    #     result = p.starmap(run_simulation, [() for i in tqdm(range(N_RUNS), desc="Time step")])

    # for run in range(N_RUNS):
    #     all_sizes.extend(result[run][0])
    #     all_times.extend(result[run][1])

    # plot_avalanches(all_times, all_sizes)

def plot_avalanches(prices): 
    log_returns = calculate_log_returns(prices)

    # Tune the threshold parameter C for wavelet filtering

    C, filtered_signal, residual_signal = tune_threshold(log_returns, wavelet='db1', level=4, target_kurtosis=3, target_skew=0, tolerance=0.1)

    # Extract avalanches from the residual signal

    avalanche_threshold = 0.01

    avalanche_sizes, avalanche_durations = extract_avalanches(residual_signal, avalanche_threshold)

    




if __name__ == "__main__":
    main()