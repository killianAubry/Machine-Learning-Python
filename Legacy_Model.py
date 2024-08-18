import numpy as np
import warnings
import csv

# Ignore all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
class Node:
    def __init__(self, input_size, is_output=False):
        self.is_output = is_output
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.input = None
        self.output = None
        self.delta = None
    def forward(self, inputs):
        self.input = inputs
        self.output = np.dot(inputs, self.weights) + self.bias
        if not self.is_output:
            self.output = self.relu(self.output)
        return self.output


    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

class NeuralNetwork:
    def __init__(self, architecture):
        self.layers = []
        self.delta_vector = []
        for i in range(1, len(architecture)):
            layer = [Node(architecture[i-1], is_output=(i == len(architecture) - 1)) for _ in range(architecture[i])]
            self.layers.append(layer)

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def forward(self, inputs):
        for layer in self.layers:
            outputs = []
            for node in layer:
                outputs.append(node.forward(inputs))
            inputs = np.array(outputs)
        return inputs

    def backward(self, error, learning_rate):
        self.delta_vector = [np.zeros(len(layer)) for layer in self.layers]
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            for j, node in enumerate(layer):
                if node.is_output:
                    node.weights += learning_rate * error * node.input
                    self.delta_vector[i][j] = error
                else:
                    error = error * self.delta_vector[i + 1] @ np.array(
                        [n.weights[j] for n in self.layers[i + 1]]) * node.relu_derivative(node.output)
                    node.weights += learning_rate * error * node.input
                    self.delta_vector[i][j] = error

    def train(self, training_data, epochs, learning_rate):
        for epoch in range(epochs):
            for inputs, target in training_data:
                inputs = np.array(inputs)
                target = np.array(target)
                output = self.sigmoid(self.forward(inputs))
                error = target - output
                self.backward(error, learning_rate)

    def predict(self, inputs):
        return self.sigmoid(self.forward(np.array(inputs)))
# Training Data: [inputs, target]
def load_and_format_data(csv_file_path):
    formatted_data = []
    with open(csv_file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data = list(map(float, row[:-1]))  # All columns except the last one
            target = [float(row[-1])]  # The last column
            formatted_data.append((data, target))
    return formatted_data

# Example usage
csv_file_path = 'data.csv'
training_data = load_and_format_data(csv_file_path)

# Define the architecture: [input_size, hidden_layer_size, output_size]
architecture = [12, 5, 3, 1]

# Initialize and train the neural network
nn = NeuralNetwork(architecture)
nn.train(training_data, epochs=1000, learning_rate=0.2)

# Test the neural network
test_cases = [
    [52,1,0,127,345,0,0,192,1,4.9,1,0],
    [62,1,0,121,357,0,1,138,0,2.8,0,0],
    [61,0,0,190,181,0,1,150,0,2.9,2,0],
    [59,0,1,190,529,1,1,151,1,3.2,2,2],
    ]

for case in test_cases:
    print(f"Input: {case}, Predicted Output: {nn.predict(case)}")
