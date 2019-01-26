# Required Python Packages
import numpy as np
import matplotlib.pyplot as plt
import pprint as pp


def sigmoid(inputs):
    """
    Calculate the sigmoid for the give inputs (array)
    :param inputs:
    :return:
    """
    return np.array([1 / float(1 + np.exp(- x)) for x in inputs])


def softmax(scores):
    """the output probabilities range is 0 to 1,
     and the sum of all the probabilities will be equal to one.

     :param inputs:  list of values
     :return: The calculated probabilities will be in the range of 0 to 1
    """
    return np.exp(scores) / np.sum(np.exp(scores), axis=0)


def tanh(input):
    # tanh(z) = 2σ(2z) − 1
    s = np.array(sigmoid(input))
    return 2*s -1


def line_graph(x, y, x_title, y_title):
    """
    Draw line graph with x and y values
    :param x:
    :param y:
    :param x_title:
    :param y_title:
    :return:
    """
    plt.axhline(0, color='gray')
    plt.axvline(0, color='gray')
    plt.plot(x, y)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()


input = range(-10, 10)
y_value = sigmoid(input)
line_graph(input, y_value, "Inputs", "Sigmoid Scores")
pp.pprint(y_value)
print('----')

input = range(-10, 10)
y_value = tanh(input)
line_graph(input, y_value, "Inputs", "Hiperbolic Tangent Scores")
pp.pprint(y_value)
print('----')


logits = range(-1, 10)
y_value = softmax(logits)
pp.pprint(y_value)
line_graph(logits, y_value, "Inputs", "Softmax Scores")



