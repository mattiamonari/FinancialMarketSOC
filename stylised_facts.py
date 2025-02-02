import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf

__all__ = ['returns_autocorrelation']

def returns_autocorrelation(returns, saveFig=False, squared=False):
    acf_values = acf(returns)

    plt.stem(acf_values)
    plt.axhline(y=-1.96 / np.sqrt(5000), color='blue', linestyle='--', label='C.I.')
    plt.axhline(y= 1.96 / np.sqrt(5000), color='blue', linestyle='--', label='C.I.')
    
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Returns Autocorrelation')

    if squared:
        plt.title('Squared Returns Autocorrelation')

    if saveFig:
        plt.savefig('returns_autocorrelation' + ('_squared_' if squared else '') + '.pdf')
        plt.cla()
    else:
        plt.show()
