import matplotlib.pyplot as plt

__all_ = ['plot_returns']

def plot_returns_vs_time(returns, saveFig=False):
    plt.plot(returns)
    plt.xlabel('Time Steps')
    plt.ylabel('Returns')
    plt.title('Price Returns')
    
    if saveFig:
        plt.savefig('returns_vs_time.pdf')
    else:
        plt.show()

def plot_returns_distribution(returns, saveFig=False):
    plt.hist(returns, bins=100)
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.title('Distribution of Returns')
    
    if saveFig:
        plt.savefig('returns_distribution.pdf')
    else:
        plt.show()

def plot_returns(returns, saveFig=False):
    plot_returns_vs_time(returns, saveFig)
    plot_returns_distribution(returns, saveFig)

def plot_market_price(prices, moving_avg, profiler_view=False, saveFig=False):
    if profiler_view:
        plt.subplot(3, 1, 1)

    plt.plot(prices, label='Market Price', color='blue')
    plt.plot(moving_avg, label='Moving Average', color='red')
    plt.xlabel('Time Steps')
    plt.ylabel('Market Price')
    plt.title('Market Price Evolution')
    plt.legend()

    if saveFig:
        plt.savefig('market_price.pdf')
    elif not profiler_view:
        plt.show()

def plot_ratio_buyers_sellers(ratio, profiler_view=False, saveFig=False):
    if profiler_view:
        plt.subplot(3, 1, 2)
    plt.plot(ratio, label='Buyers/Sellers', color='green')
    plt.xlabel('Time Steps')
    plt.ylabel('Ratio')
    plt.title('Buyers/Sellers Ratio')
    plt.legend()
    plt.tight_layout()

    if saveFig:
        plt.savefig('ratio_buyers_sellers.pdf')
    elif not profiler_view:
        plt.show()

def plot_weighted_volumes(weighted_volumes, profiler_view=False, saveFig=False):
    if profiler_view:
        plt.subplot(3, 1, 3)
    
    plt.plot(weighted_volumes)
    plt.ylim(0, 1)
    plt.xlabel('Time Steps')
    plt.ylabel('Volumes Proportion (Buy)')
    
    if saveFig:
        plt.savefig('weighted_volumes.pdf')
    else: # In this case since this is the last plot we show it even if profileview is True
        plt.show()