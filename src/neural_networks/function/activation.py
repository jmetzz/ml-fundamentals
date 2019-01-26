import math
import numpy as np


def step(x):
    return 1.0 if x >= 0.0 else 0.0


def sigmoid(x):
    """the sigmoid function take any range real number
    and returns the output value which falls in the range of 0 to 1."""
    return 1 / (1 + math.exp(-x))


def hypertang(x):
    exp = math.exp(2 * x)
    return (exp - 1) / (exp + 1)


def softmax(inputs):
    """the output probabilities range is 0 to 1,
     and the sum of all the probabilities will be equal to one.

     param inputs:  list of values
     return: The calculated probabilities will be in the range of 0 to 1
    """
    return np.exp(inputs) / float(sum(np.exp(inputs)))
