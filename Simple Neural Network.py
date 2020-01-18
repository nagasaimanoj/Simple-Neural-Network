from numpy import (
    array,  # for numpy arrays
    dot,  # for matrix multiplication
    exp,  # used to find sigmoid function
    random  # for random number generation
)


class SimpleNeuralNetwork:
    def __init__(self, dimensions):
        """ initializing weights based on dimensions count """
        weights = list()
        for i in range(dimensions):
            weights.append([random.random()])
        self.weights = array(weights)

    @staticmethod
    def sigmoid_func(val):
        """ Sigmoid Function """
        return 1 / (1 + exp(-val))

    def predict(self, pred_vals):
        """ sigmoid(sum(each point * its weight)) """
        return self.sigmoid_func(dot(pred_vals, self.weights))

    def train(self, input_x, input_y, iterations):
        """ predicting, finding error, updating weights, start over """
        for _ in range(iterations):
            # predicting with current weights
            prediction = self.predict(input_x)

            # difference in actual output and current output
            error = input_y - prediction

            # updating weights with gradient based on error value
            self.weights += dot(input_x.T, error * prediction * (1 - prediction))


if __name__ == '__main__':
    """ simple example usecase for above module """
    # independent variables for training
    train_x = array([
        [0, 0], [1, 1], [1, 0], [0, 1]
    ])
    # dependent variables for training
    train_y = array([
        [0], [1], [1], [0]
    ])

    nn = SimpleNeuralNetwork(dimensions=2)  # initialise neural network model with random weights
    nn.train(train_x, train_y, 100)  # train the models (update weights based on independent & dependent variables)

    test_x = array([1, 0])  # independent variables for testing
    test_y = nn.predict(test_x)  # predicting dependent variables from independent variables
    print(test_x, '=>', test_y)
