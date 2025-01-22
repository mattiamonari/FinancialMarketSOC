import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Parameters for the simulation
NUM_NODES = 1000  # Total number of traders (including hedge funds)
NUM_HEDGE_FUNDS = 10  # Number of hedge funds (high-degree nodes)
ALPHA = 0.7  # Weight for trade size influence
BETA = 0.3  # Weight for degree influence
GAMMA = 1  # Sensitivity for profit acceptance
ETA = 0.01  # Scaling factor for price changes
TIME_STEPS = 1000  # Number of time steps for the simulation

# Initialize a scale-free network using Barabási-Albert model
G = nx.barabasi_albert_graph(NUM_NODES, m=5)

# Assign node attributes (traders and hedge funds)
for node in G.nodes:
    if node < NUM_HEDGE_FUNDS:
        G.nodes[node]['type'] = 'hedge_fund'
        G.nodes[node]['profit_threshold'] = np.random.normal(0.3, 0.1)  # Example profit threshold
        G.nodes[node]['trade_size'] = np.random.uniform(1, 2)  # Random trade size
    else:
        G.nodes[node]['type'] = 'trader'
        G.nodes[node]['profit_threshold'] = np.random.normal(0.3, 0.1)  # Example profit threshold (not used at the moment)
        G.nodes[node]['trade_size'] = np.random.uniform(0.2, 0.6)  # Random trade size

    G.nodes[node]['last_update_time'] = 0
    G.nodes[node]['position'] = 'buy' if np.random.random() < 0.5 else 'sell'

# Initialize market price
price = 100
prices = [price]

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
                if np.random.rand() < influence and G.nodes[node]['last_update_time'] < t - 10:
                    G.nodes[node]['position'] = G.nodes[neighbor]['position']
                    G.nodes[node]['last_update_time'] = t

# Function to update market price
def update_price():
    global price
    buy_volume = sum(G.nodes[node]['trade_size'] for node in G.nodes if G.nodes[node]['position'] == 'buy')
    sell_volume = sum(G.nodes[node]['trade_size'] for node in G.nodes if G.nodes[node]['position'] == 'sell')
    print("Buy Volume: ", buy_volume, "Sell Volume: ", sell_volume)
    price += ETA * (buy_volume - sell_volume)
    prices.append(price)

# Run the simulation
for t in range(TIME_STEPS):
    update_positions(t)
    update_price()

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