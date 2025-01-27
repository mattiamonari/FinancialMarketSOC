import numpy as np

def lag_check(t, t_update, max_lag = 5, min_lag = 1):
    if t - t_update > np.random.randint(min_lag, max_lag):
        return True
    return False