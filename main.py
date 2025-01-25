import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from market_statistics import * 
from plots import *

# Parameters for the simulation
NUM_NODES = 1000  # Total number of traders (including hedge funds)
NUM_HEDGE_FUNDS = 10  # Number of hedge funds (high-degree nodes)
ALPHA = 0.05  # Weight for trade size influence
BETA = 0.1  # Weight for degree influence
GAMMA = 1  # Sensitivity for profit acceptance
ETA = 0.01  # Scaling factor for price changes
EXPONENTIAL_SCALING = 1.1 # Exponential scaling factor for high volume differences
TIME_STEPS = 1000  # Number of time steps for the simulation

# Initialize a scale-free network using Barabási-Albert model
G = nx.barabasi_albert_graph(NUM_NODES, m=NUM_HEDGE_FUNDS)

# Initialize lists to track the number of buyers and sellers
num_buyers = []
num_sellers = []

# Order nodes by degree and assign attributes. This is done to ensure that hedge funds are high-degree nodes.
sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)  # Sort nodes by degree
for idx, (node, degree) in enumerate(sorted_nodes):
    if idx < NUM_HEDGE_FUNDS:
        G.nodes[node]['type'] = 'hedge_fund'
        G.nodes[node]['profit_threshold'] = np.random.normal(0.5, 0.1)  # Example profit threshold
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
            if np.random.rand() < avg_influence:
                influential_neighbor = max(neighbors, key=lambda n: G.degree[n])
                G.nodes[node]['position'] = G.nodes[influential_neighbor]['position']
                G.nodes[node]['last_update_time'] = t
                G.nodes[node]['trade_size'] = np.random.uniform(0.2, 1)  # Random trade size

            # Traders are influenced by neighbors
            # for neighbor in neighbors:              
            #     influence = ALPHA * G.nodes[neighbor]['trade_size'] / 10 + BETA * len(neighbors)
            #     if np.random.rand() < influence and G.nodes[node]['last_update_time'] < t - random_lags[t]:
            #         G.nodes[node]['position'] = G.nodes[neighbor]['position']
            #         G.nodes[node]['last_update_time'] = t

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
    
    volume_difference = abs(buy_volume - sell_volume)
    # Apply exponential scaling for large volume differences
    if volume_difference > (buy_volume + sell_volume) / 2:  # Example threshold
        price += ETA * np.sign(buy_volume - sell_volume) * (volume_difference ** EXPONENTIAL_SCALING)
    else:
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




