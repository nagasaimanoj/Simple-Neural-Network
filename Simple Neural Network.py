from numpy import array, dot, exp


def sigmoid_func(val):
    return 1 / (1 + exp(-val))


input_set = array([
    [0, 0],
    [1, 1],
    [1, 0],
    [0, 1]
])

output_set = array([
    [0],
    [1],
    [1],
    [0]
])

weights = array([
    [0.],
    [0.]
])

for iteration in range(100):
    mat_mul = dot(input_set, weights)
    prediction = sigmoid_func(mat_mul)

    error = output_set - prediction

    weights += dot(
        input_set.T,
        error * prediction * (1 - prediction)
    )

print("[1, 0] =", sigmoid_func(
    dot(array([1, 0]), weights)
))
