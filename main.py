import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from market_statistics import * 
from plots import *

# Parameters for the simulation
NUM_NODES = 1000  # Total number of traders (including hedge funds)
NUM_HEDGE_FUNDS = 10  # Number of hedge funds (high-degree nodes)
ALPHA = 0.2  # Weight for trade size influence
BETA = 0.1  # Weight for degree influence
GAMMA = 1  # Sensitivity for profit acceptance
ETA = 0.01  # Scaling factor for price changes
TIME_STEPS = 1000  # Number of time steps for the simulation

# Initialize a scale-free network using Barabási-Albert model
G = nx.barabasi_albert_graph(NUM_NODES, m=NUM_HEDGE_FUNDS)

# Initialize lists to track the number of buyers and sellers
num_buyers = []
num_sellers = []

# Assign node attributes (traders and hedge funds)
for node in G.nodes:
    if node < NUM_HEDGE_FUNDS:
        G.nodes[node]['type'] = 'hedge_fund'
        G.nodes[node]['profit_threshold'] = np.random.normal(0.3, 0.1)  # Example profit threshold
        G.nodes[node]['trade_size'] = np.random.uniform(10, 50)  # Random trade size
    else:
        G.nodes[node]['type'] = 'trader'
        G.nodes[node]['profit_threshold'] = np.random.normal(0.3, 0.1)  # Example profit threshold (not used at the moment)
        G.nodes[node]['trade_size'] = np.random.uniform(0.2, 1)  # Random trade size

    G.nodes[node]['last_update_time'] = 0
    G.nodes[node]['position'] = 'buy' if np.random.random() < 0.5 else 'sell'

# Initialize market price
price = 1000
prices = [price]
random_lags = np.random.randint(1, 5, size=TIME_STEPS)

# Function to update positions
def update_positions(t):
    global price
    global random_lags
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
                if np.random.rand() < influence and G.nodes[node]['last_update_time'] < t - random_lags[t]:
                    G.nodes[node]['position'] = G.nodes[neighbor]['position']
                    G.nodes[node]['last_update_time'] = t

# Function to update market price
def update_price():
    global price
    buy_volume = sum(G.nodes[node]['trade_size'] for node in G.nodes if G.nodes[node]['position'] == 'buy')
    sell_volume = sum(G.nodes[node]['trade_size'] for node in G.nodes if G.nodes[node]['position'] == 'sell')

    # Count the number of buyers and sellers
    buyers = sum(1 for node in G.nodes if G.nodes[node]['position'] == 'buy')
    sellers = sum(1 for node in G.nodes if G.nodes[node]['position'] == 'sell')
    
    num_buyers.append(buyers)
    num_sellers.append(sellers)
    
    price += ETA * (buy_volume - sell_volume)
    prices.append(price)

weighted_volumes = np.zeros(TIME_STEPS)

# Run the simulation
for t in tqdm(range(TIME_STEPS), desc="Time Step"):
    update_positions(t)
    update_price()
    weighted_volumes[t] = calculate_volume_imbalance(G)

# Compute the moving average of the market price
moving_avg = np.convolve(prices, np.ones(25) / 25, mode='valid')
print("Market volatility: ", np.std(prices))

returns = calculate_price_returns(prices)
print("The mean of returns is: ", np.mean(returns))
plot_returns(returns, saveFig=False)

# Added for the 'profile view' in the plot. If in the future we remove it, change the method parameter and remove this.
plt.figure(figsize=(15, 6))
plot_market_price(prices, moving_avg, profiler_view=True, saveFig=False)
ratio = [num_buyers[i] / (num_sellers[i] + num_buyers[i]) for i in range(len(num_buyers))]
plot_ratio_buyers_sellers(ratio, profiler_view=True, saveFig=False)
plot_weighted_volumes(weighted_volumes, profiler_view=True, saveFig=False)




