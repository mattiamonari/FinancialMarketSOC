
__all_ = ['calculate_price_returns']

def calculate_price_returns(prices):
    returns = []
    for i in range(len(prices) - 1):
        returns.append((prices[i + 1] - prices[i]) / prices[i])
    return returns