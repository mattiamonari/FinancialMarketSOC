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