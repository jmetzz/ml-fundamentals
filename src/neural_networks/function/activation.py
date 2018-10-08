import math


def step(x):
    return 1.0 if x >= 0.0 else 0.0


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def hypertang(x):
    exp = math.exp(2 * x)
    return (exp - 1) / (exp + 1)