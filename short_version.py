from numpy import array, dot, exp

training_set_inputs = array([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
])

training_set_outputs = array([
    [0],
    [1],
    [1],
    [0]
])

synaptic_weights = array([
    [0.],
    [0.],
    [0.]
])

for iteration in range(10000):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))

    synaptic_weights += dot(training_set_inputs.T,
                            (training_set_outputs - output) * output * (1 - output))

print(1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights)))))
