import numpy as np
import warnings
import csv
import json
import multiprocessing as mp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Ignore all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
training_loss = []
time_per_epoch = []
f1_scores = {
    "true_positive": 0,
    "true_negative": 0,
    "false_positive": 0,
    "false_negative": 0
}

def progress_bar(total_steps, current_step):
    bar_length = 50
    filled_length = int(bar_length * current_step / total_steps)
    bar = "=" * filled_length + " " * (bar_length - filled_length)
    print(f"[{bar}] {current_step}/{total_steps}", end="\r")


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

    def sigmoid_derivative(self, x):
        return x * (1 - x)

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
        if x<-700:
            return 0
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)
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
                    node.bias += learning_rate * error
                    self.delta_vector[i][j] = error

    def train(self, training_data, epochs, learning_rate):
        global training_loss, time_per_epoch, f1_scores
        training_data_len = len(training_data)
        for epoch in range(epochs):
            epoch_loss = 0
            for inputs, target in training_data:
                inputs = np.array(inputs)
                target = np.array(target)
                output = self.sigmoid(self.forward(inputs))
                f1_scores["true_positive"] += 1 if output >= 0.5 and target == 1 else 0
                f1_scores["true_negative"] += 1 if output < 0.5 and target == 0 else 0
                f1_scores["false_positive"] += 1 if output >= 0.5 and target == 0 else 0
                f1_scores["false_negative"] += 1 if output < 0.5 and target == 1 else 0
                error = target - output
                self.backward(error, learning_rate)

            # Update the plot every 100 epochs
            if (epoch) % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, % Correct: {100*((f1_scores["true_positive"]+f1_scores["false_negative"]) / training_data_len)}%")
                animate_predictions(self, training_data, title=f'Epoch {epoch + 1}/{epochs}')
                # animate_predictions(self, training_data, ax, title=f'Epoch {epoch + 1}/{epochs}'
                print("next")
            f1_scores = {
                "true_positive": 0,
                "true_negative": 0,
                "false_positive": 0,
                "false_negative": 0
            }


    def predict(self, inputs):
        return self.sigmoid(self.forward(inputs))


def generate_report():
    global training_loss, time_per_epoch, f1_scores
    process_data = {
        'training_loss': training_loss,
        'time_per_epoch': time_per_epoch,
        'f1_score': f1_scores,
    }
    file_path = 'process1.json'
    with open(file_path, 'w') as file:
        json.dump(process_data, file)


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


def animate_predictions(nn, training_data, title='Neural Network Predictions', f1_scores=False):
    group1 = []
    group2 = []
    test_cases = [
        [3.375698018345672, 2.3993613100811952, 2.708306250206723,1],
        [3.4040508568145382, 4.88618590121053, 3.174577812831839,1],
        [7.216458589581975, 7.045571839903814, 6.348399652394183,0],
        [7.651391251305798, 6.684730755359654, 7.758969220493268,0]
    ]

    for case in test_cases:
        print(f"Input: {case}, Predicted Output: {nn.predict(case[:-1])} Actual Output: {case[-1]}")

    for inputs, _ in training_data:
        prediction = nn.predict(inputs)
        if prediction >= 0.5:
            group1.append(inputs)
        else:
            group2.append(inputs)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.cla()
    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)

    if group1:
        group1 = np.array(group1)
        ax.scatter(group1[:, 0], group1[:, 1], group1[:, 2], color='blue', label='Group 1')
    if group2:
        group2 = np.array(group2)
        ax.scatter(group2[:, 0], group2[:, 1], group2[:, 2], color='red', label='Group 2')

    ax.legend()
    plt.draw()
    plt.show()


# Example usage
csv_file_path = 'data.csv'
training_data = load_and_format_data(csv_file_path)

# Define the architecture: [input_size, hidden_layer_size, output_size]
architecture = [3, 50, 50, 50, 1]

# Initialize and train the neural network
nn = NeuralNetwork(architecture)
nn.train(training_data, epochs=1000, learning_rate=0.03)
# Test the neural network
test_cases = [
    [3.375698018345672,2.3993613100811952,2.708306250206723],
    [3.4040508568145382,4.88618590121053,3.174577812831839],
    [7.216458589581975,7.045571839903814,6.348399652394183],
    [7.651391251305798,6.684730755359654,7.758969220493268]
]

for case in test_cases:
    print(f"Input: {case}, Predicted Output: {nn.predict(case)}")
