import numpy as np
import pandas as pd
import scipy as sc
from scipy import stats


# Input a pandas series
def ent(data):
    p_data = data.value_counts() / len(data)  # calculates the probabilities
    entropy = sc.stats.entropy(p_data, base=2)  # input probabilities to get the entropy
    # entropy = sc.stats.entropy(p_data)  # input probabilities to get the entropy
    return entropy


data = pd.Series(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]))
print(ent(data))

data = pd.Series(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
print(ent(data))
