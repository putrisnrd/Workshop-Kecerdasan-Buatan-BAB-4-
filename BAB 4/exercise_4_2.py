# Example 4.2 Multiple Layer Perceptron (Modified for 3 Inputs)
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html
from sklearn.neural_network import MLPClassifier

# Updated input feature matrix X with 3 features per sample (8 binary combinations)
X = [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
]

# Labels (e.g., 3-input logical OR: 0 only when all inputs are 0, else 1)
y = [0, 1, 1, 1, 1, 1, 1, 1]

# Initialize MLP with hidden layers of sizes 5 and 2
clf = MLPClassifier(
    solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
)

# Train the model
clf.fit(X, y)

# Predict on new 3-input test samples
print("Predictions:", clf.predict([[2.0, 2.0, 2.0], [-1.0, -2.0, 0.0]]))

# Print the weight matrix shapes at each layer
print("Coefficient matrix shapes:", [coef.shape for coef in clf.coefs_])