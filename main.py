import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
import powerlaw


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
                if np.random.rand() < influence and G.nodes[node]['last_update_time'] < t - 10:
                    G.nodes[node]['position'] = G.nodes[neighbor]['position']
                    G.nodes[node]['last_update_time'] = t

# Function to update market price
def update_price(t):
    global price
    buy_volume = sum(G.nodes[node]['trade_size'] for node in G.nodes if G.nodes[node]['position'] == 'buy')
    sell_volume = sum(G.nodes[node]['trade_size'] for node in G.nodes if G.nodes[node]['position'] == 'sell')
    #print("Buy Volume: ", buy_volume, "Sell Volume: ", sell_volume)
    price += ETA * (buy_volume - sell_volume)
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





