import math


def step(x):
    return 1.0 if x >= 0.0 else 0.0


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def hypertang(x):
    exp = math.exp(2 * x)
    return (exp - 1) / (exp + 1)


class Perceptron:

    def __init__(self, activation_function=step):
        self.activation = activation_function

    def predict(self, x, w):
        s = self._summation(x, w)
        return self.activation(s)

    def _summation(self, x, w):
        # The first weight is always the bias as it is standalone
        # and not responsible for a specific input value.
        activation = w[0]
        for i in range(len(w) - 1):
            activation += w[i + 1] * x[i]
        return activation


class SingleLayerPerceptron:
    def __init__(self, bias, dimension, epochs=50, learning_rate=0.001, activation_function=step):
        self.neuron = Perceptron(activation_function)
        self.bias = bias
        self.dimension = dimension
        self.eta = learning_rate
        self.epochs = epochs

    def train(self, data):
        weights = [0.0] * (self.dimension + 1)
        for e in range(self.epochs):
            sum_error = 0.0
            for x in data:
                # print(weights) # uncomment the check the change in weights
                predicted = self.neuron.predict(x[:-1], weights)
                error = x[-1] - predicted
                sum_error += (error) ** 2
                for i in range(self.dimension):
                    weights[i + 1] += self.eta * error * x[i]
            print('>epoch=%d, learning rate=%.3f, error=%.3f' % (e, self.eta, sum_error))
        return weights


# test predictions
dataset = [[2.7810836, 2.550537003, 0],
           [1.465489372, 2.362125076, 0],
           [3.396561688, 4.400293529, 0],
           [1.38807019, 1.850220317, 0],
           [3.06407232, 3.005305973, 0],
           [7.627531214, 2.759262235, 1],
           [5.332441248, 2.088626775, 1],
           [6.922596716, 1.77106367, 1],
           [8.675418651, -0.242068655, 1],
           [7.673756466, 3.508563011, 1]]


def main():
    neuron = Perceptron()
    # The values of bias and weights are typically set randomly and then updated using gradient descent
    w = [-0.1, 0.20653640140000007, -0.23418117710000003]
    for row in dataset:
        prediction = neuron.predict(row, w)
        print("Expected=%d, Predicted=%d" % (row[-1], prediction))

    print("-----------------------------")
    rna = SingleLayerPerceptron(-0.1, 2, 50, 0.01, sigmoid)

    nweights = rna.train(dataset)
    print("Network weights: ")
    print(nweights)


if __name__ == '__main__':
    main()
