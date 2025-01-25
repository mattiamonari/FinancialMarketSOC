
__all_ = ['calculate_price_returns', 'calculate_volume_imbalance']

def calculate_price_returns(prices):
    returns = []
    for i in range(len(prices) - 1):
        if prices[i] == 0:
            returns.append(0)
            continue
        returns.append((prices[i + 1] - prices[i]) / prices[i])
    return returns

def calculate_volume_imbalance(G):
    buy_volume = sum(G.nodes[node]['trade_size'] for node in G.nodes if G.nodes[node]['position'] == 'buy')
    sell_volume = sum(G.nodes[node]['trade_size'] for node in G.nodes if G.nodes[node]['position'] == 'sell')

    return buy_volume / (sell_volume + buy_volume)