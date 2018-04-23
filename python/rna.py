class Perceptron:

    def __init__(self, bias, dimension):
        self.bias = bias
        self.dimension = dimension

    def predict(self, x, w):
        return 1.0 if self._activation(x, w) >= 0.0 else 0.0

    def _activation(self, x, w):
        # The first weight is always the bias as it is standalone
        # and not responsible for a specific input value.
        activation = w[0]
        for i in range(self.dimension):
            activation += w[i + 1] * x[i]
        return activation


class SingleLayerPerceptron:
    def __init__(self, bias, dimension, epochs=50, learning_rate=0.001):
        self.neuron = Perceptron(bias, dimension)
        self.dimension = dimension
        self.eta = learning_rate
        self.epochs = epochs

    def train(self, data):
        weights = [0.0 for i in range(self.dimension + 1)]
        for e in range(self.epochs):
            sum_error = 0.0
            for x in data:
                predicted = self.neuron.predict(x[:-1], weights)
                error = x[-1] - predicted
                sum_error += (error) ** 2
                for i in range(self.dimension):
                    weights[i + 1] += self.eta * error * x[i]
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (e, self.eta, sum_error))
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
    p = Perceptron(0.5, 2)
    w = [-0.1, 0.20653640140000007, -0.23418117710000003]
    for row in dataset:
        prediction = p.predict(row, w)
        print("Expected=%d, Predicted=%d" % (row[-1], prediction))

    print("-----------------------------")
    rna = SingleLayerPerceptron(-0.1, 2, 10, 0.1)

    nweights = rna.train(dataset)
    print("Network weights: ")
    print(nweights)


if __name__ == '__main__':
    main()
