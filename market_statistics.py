import numpy as np


__all_ = ['calculate_price_returns', 'calculate_volume_imbalance']

def calculate_price_returns(prices):
    returns = []
    for i in range(len(prices) - 1):
        if prices[i] == 0:
            returns.append(0)
            continue
        returns.append((prices[i + 1] - prices[i]) / prices[i])
    return np.array(returns)

def calculate_volume_imbalance(G):
    buy_volume = sum(G.nodes[node]['trade_size'] for node in G.nodes if G.nodes[node]['position'] == 'buy')
    sell_volume = sum(G.nodes[node]['trade_size'] for node in G.nodes if G.nodes[node]['position'] == 'sell')

    return buy_volume / (sell_volume + buy_volume)

def detect_avalanche(moving_avg, threshold_start=0.0015, threshold_end=0.01):
    avalanche_starts = []
    avalanche_ends = []
    avalanche_times = []
    i = 1
    
    while i < len(moving_avg) - 1:
        # Check for avalanche start
        change = (moving_avg[i + 1] - moving_avg[i]) / moving_avg[i]
        
        if abs(change) > threshold_start:
            # Mark avalanche start
            
            start = i
            end = -1
            for j in range(i + 1, len(moving_avg) - 1):
                end_change = (moving_avg[j + 1] - moving_avg[j]) / moving_avg[j]
                
                if abs(end_change) < threshold_end:
                    avalanche_times.append(j - i)
                    end = j
                    
                    # Move index to the end of this avalanche
                    i = j
                    break

            if end != -1:
                avalanche_starts.append(start)
                avalanche_ends.append(end)

        # Increment to next time step
        i += 1
    
    return avalanche_starts, avalanche_ends, avalanche_times