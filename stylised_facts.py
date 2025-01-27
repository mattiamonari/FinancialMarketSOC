import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf

__all__ = ['returns_autocorrelation']

def returns_autocorrelation(returns, saveFig=False):
    acf_values = acf(returns)
    plt.stem(acf_values)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Returns Autocorrelation')
    
    if saveFig:
        plt.savefig('returns_autocorrelation.pdf')
    else:
        plt.show()
