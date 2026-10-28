# Example 4.1 Neuron.py (Modified for Logical OR with Bias)
from numpy import array, dot, exp, random

# Set up parameters ===============================================
# Added a bias feature (1.0) as the third column in inputs
inputs = array(
    [
        [0, 0, 1],  # [Input1, Input2, Bias] -> 0 OR 0 = 0
        [1, 1, 1],  # [Input1, Input2, Bias] -> 1 OR 1 = 1
        [1, 0, 1],  # [Input1, Input2, Bias] -> 1 OR 0 = 1
        [0, 1, 1],  # [Input1, Input2, Bias] -> 0 OR 1 = 1
    ]
)

# Target OR outputs
outputs = array([[0, 1, 1, 1]]).T

random.seed(1)
# Initialize weights for 2 inputs + 1 bias
weights = 2 * random.random((3, 1)) - 1


# Define Single Neuron function ===================================
def neuron(inputs, weights):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    return output


# Train the Neuron ================================================
for iteration in range(50000):
    output = neuron(inputs, weights)
    # Derivative of Sigmoid: output * (1 - output)
    error = outputs - output
    weights += dot(inputs.T, error * output * (1 - output))

# Test the Neuron =================================================
# Test case: Input (1, 0) + Bias (1)
x = array([1, 0, 1])
print("Prediction for [1, 0]:", neuron(x, weights))

# Additional Test Cases
print("Prediction for [0, 0]:", neuron(array([0, 0, 1]), weights))
print("Prediction for [1, 1]:", neuron(array([1, 1, 1]), weights))
print("Prediction for [0, 1]:", neuron(array([0, 1, 1]), weights))