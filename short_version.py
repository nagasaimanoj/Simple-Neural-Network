from numpy import array, dot, exp

input_set = array([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
])

output_set = array([
    [0],
    [1],
    [1],
    [0]
])

weights = array([
    [0.],
    [0.],
    [0.]
])

for iteration in range(10000):
    output = 1 / (1 + exp(-(dot(input_set, weights))))

    weights += dot(input_set.T,
                            (output_set - output) * output * (1 - output))

print(1 / (1 + exp(-(dot(array([1, 0, 0]), weights)))))
