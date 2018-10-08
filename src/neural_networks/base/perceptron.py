


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
