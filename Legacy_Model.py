import numpy as np
import random
import pandas as pd
import time
import json 
import multiprocessing
import matplotlib.pyplot as plt
plt.ion()
class Node:
   def __init__(self, input_size, mu, is_output=False):
       self.mu = mu
       self.bias =  np.random.uniform(-1, 1)
       self.is_output = is_output
       self.input_size = input_size
       self.initialize_weights_xavier()
   def initialize_weights_xavier(self):
       variance = 2 / (self.input_size + 1)  # Add 1 for the bias
       self.w = np.random.normal(0, np.sqrt(variance), size=self.input_size + 1)
   def dotProduct(self, x):
       self.x = x
       raw_output = np.dot(x, self.w[:-1]) + (self.w[-1]*self.bias) # Legacy Opperation
       if not self.is_output:
           return self.relu(raw_output)
       else:
           return raw_output
   def error(self, percentError):
       for j in range(len(self.w) - 1):
           self.w[j] = self.w[j] + self.mu * (percentError) * self.x[j]
       self.w[-1] += self.mu * (percentError)
   def sigmoid(self, input):
       return 1 / (1 + np.exp(-(input)))
   def relu(self, input):
       return np.maximum(0.1 * input, input)
   def tanh(self, input):
       return np.tanh(input)
   def normTanh(self, input):
       return (np.tanh(input) + 1) / 2
   def mean_squared_error(self, predictions, targets):
       return np.mean((predictions - targets) ** 2)
def create_nodes(architecture, learningRate):
   nodes = []
   for i in range(1, len(architecture)):
       for j in range(architecture[i]):
           is_output = i == len(architecture) - 1
           input_size = architecture[i - 1]
           nodes.append(Node(input_size, learningRate, is_output))
   return nodes
def feedForward(architecture, nodes, inputs):
   layer_inputs = inputs
   node_index = 0
   for layer_size in architecture[1:]:
       layer_outputs = []
       for _ in range(layer_size):
           node_output = nodes[node_index].dotProduct(layer_inputs)
           layer_outputs.append(node_output)
           node_index += 1
       layer_inputs = layer_outputs
   return layer_inputs
def plot_decision_boundary(nodes, x_range, y_range, terminate, architecture):
   if terminate:
       plt.clf()
   x_vals = np.linspace(x_range[0], x_range[1], 50)
   y_vals = np.linspace(y_range[0], y_range[1], 50)
   decision_boundary = np.zeros((len(x_vals), len(y_vals)))
   for i, x in enumerate(x_vals):
       for j, y in enumerate(y_vals):
           nFR = feedForward(architecture, nodes, [x, y, x**2, y**2])
           decision_boundary[i, j] = nodes[len(nodes)-1].normTanh((nFR))
   #print(decision_boundary)
   plt.contourf(x_vals, y_vals, decision_boundary.T, [0, 0.5, 1], cmap='coolwarm', alpha=0.8)
   x_values = [point[0] for point in trainingData]
   y_values = [point[1] for point in trainingData]
   labels = [point[2] for point in trainingData]
   for i in range(len(trainingData)):
       color = 'red' if labels[i] == 1 else 'blue'
       plt.scatter(x_values[i], y_values[i], color=color)
   plt.xlabel('X-axis')
   plt.ylabel('Y-axis')
   plt.title('Decision Boundary')
   plt.pause(0.000000001)
   if not terminate:
       plt.show(block=True)
   else:
        plt.show()
def print_progress_bar(percentage):
   bar_length = 30
   num_blocks = int(round(bar_length * percentage))
   progress = "[" + "█" * num_blocks + "-" * (bar_length - num_blocks) + "]"
   percentage_str = f"{percentage * 100:.2f}%"
   print(f"\r{progress.ljust(bar_length + 2)}{percentage_str.ljust(6)}", end="")
   if percentage == 1.0:
       print()  # Move to the next line after the progress bar is complete

class NeuralNetwork:
    def __init__(self, architecture, mu, trainingData, iteration):
       self.mu = mu
       self.architecture = architecture
       self.trainingData = trainingData
       self.trainMonitors = []
       self.timePerEpoch = []
       self.nodes = create_nodes(self.architecture, self.mu)
       self.pred = []
       self.true_positives = 0
       self.false_positives = 0
       self.false_negatives = 0
       self.f1_score = 0
       self.true_negatives = 0
       local_seed = iteration
       np.random.seed(local_seed)
       self.seed = local_seed

    
    def train(self, epochs, reports):
        startTime = time.time()
        for epoch in range(epochs):
            print_progress_bar(epoch/epochs)
            if reports and epoch%reports==0:
                plot_decision_boundary(self.nodes, [-4, 11], [-4, 11], True,self.architecture)
            avg_loss = 0
            for i in trainingData:
                xsq = i[0]*i[0]
                x2sq = i[1]*i[1]
                final = i[:-1]
                final.append(xsq)
                final.append(x2sq)
                nFR = feedForward(self.architecture, self.nodes, final)
                NormalizedNFR = self.nodes[-1].normTanh(nFR)
                loss = i[len(i) - 1] - NormalizedNFR
                avg_loss += loss
                self.pred = NormalizedNFR
                self.true_positives += (i[-1] == 1) and (self.pred == 1)
                self.false_positives += (i[-1] == 0) and (self.pred == 1)
                self.false_negatives += (i[-1] == 1) and (self.pred == 0)
                self.true_negatives += (i[-1] == 0) and (self.pred == 0)
                for j in range(len(self.nodes)-1):
                    self.nodes[j].error(loss)
            avg_loss/=len(trainingData)
            self.trainMonitors.append(abs(avg_loss[-1]))
        endTime = time.time()
        self.timePerEpoch.append(endTime - startTime)

    def testCase(self, x,y):
        xsq = x**2
        x2sq = y**2
        final = [xsq, x2sq]
        final.append(xsq*2)
        final.append(x2sq*2)
        nFR = feedForward(self.architecture, self.nodes, final)
        print("Test Case:", [x,y], " --> ", self.nodes[-1].normTanh((nFR)))
        plot_decision_boundary(self.nodes, [-4, 11], [-4, 11], True, self.architecture)
    def plot(self, bounds):
        plot_decision_boundary(self.nodes, bounds, bounds, True, self.architecture)
    def trainingLoss(self, disp):
        plt.clf()
        x_values = np.linspace(0, len(self.trainMonitors)-1, len(self.trainMonitors))
        y_values = [point for point in self.trainMonitors]
        # Convert data to pandas DataFrame for easier manipulation
        df = pd.DataFrame({'Epochs': x_values, 'Loss': y_values})

        # Calculate the exponential moving average (EMA) with a smoothing factor (adjust alpha as needed)
        df['EMA'] = df['Loss'].ewm(alpha=0.05).mean()

        # Plot original data
        plt.plot(df['Epochs'], df['Loss'], label='Original Loss', linestyle=':')

        # Plot EMA data
        plt.plot(df['Epochs'], df['EMA'], label='EMA', linestyle='-')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('error with EMA')
        plt.legend()  # Display legend
        plt.grid(True)
        
        # Set the bottom limit of the y-axis to 0
        #plt.ylim(bottom=0)

        plt.ioff()
        plt.show()
    def AverageTimePerEpoch(self):
        print()
        print("__________________________________________________________")
        print("Averge Time Per Epoch:",(sum(self.timePerEpoch))/(len(self.timePerEpoch)), "Seconds")
    def calculate_f1_score(self, true_positives, false_positives, false_negatives):
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
        self.f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        print()
        print("__________________________________________________________")
        print("F1 Score:", self.f1_score)
    def plot_f1_score(self): 
        total_samples = self.true_negatives + self.false_positives + self.false_negatives + self.true_positives

        confusion_matrix = np.array([[self.true_negatives, self.false_positives], [self.false_negatives, self.true_positives]]) / total_samples * 100
        fig, ax = plt.subplots()
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="7%", pad="2%")
        
        # Use the same colormap and normalization as in the provided code
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(0, 100)  # Normalize to percentage scale

        im = ax.imshow(confusion_matrix, cmap=cmap, norm=norm)
        cb = fig.colorbar(im, cax=cax)
        trNeg = np.round(self.true_negatives/ total_samples * 100, 1)
        faPos = np.round(self.false_positives/ total_samples * 100, 1)
        faNeg = np.round(self.false_negatives/ total_samples * 100, 1)
        trPos = np.round(self.true_positives/ total_samples * 100, 1)
        ax.text(0, 0, f"TN: {trNeg}%", ha='center', va='center', color='white' if trNeg < 50 else 'black', fontsize=15)
        ax.text(1, 0, f"FP: {faPos}%", ha='center', va='center', color='white' if faPos < 50 else 'black', fontsize=15)
        ax.text(0, 1, f"FN: {faNeg}%", ha='center', va='center', color='white' if faNeg < 50 else 'black', fontsize=15)
        ax.text(1, 1, f"TP: {trPos}%", ha='center', va='center', color='white' if trPos < 50 else 'black', fontsize=15)
        ax.set_title('Confusion Matrix (Percentages)')
        ax.set_axis_off()
        plt.ioff()
        plt.show()
    def log_results(self, log_filename):
        # Convert NumPy arrays to Python lists within training_loss
        local_trainMonitors = (self.trainMonitors)
        local_timePerEpoch = (self.timePerEpoch)
        results = {
            "architecture": self.architecture,
            "marker_type": "legacy",
            "seed": self.seed,
            "learning_rate": float(self.mu),
            "training_loss": local_trainMonitors,
            "average_time_per_epoch": sum(local_timePerEpoch) / len(local_timePerEpoch),
            "f1_score": float(self.f1_score),
            "confusion_matrix": {
                "true_negatives": int(self.true_negatives),
                "false_positives": int(self.false_positives),
                "false_negatives": int(self.false_negatives),
                "true_positives": int(self.true_positives),
            }
        }
        with open(log_filename, 'w') as log_file:
            json.dump(results, log_file)







trainingData =  [[0.07056451612903225, 0.1515151515151515, 1], [0.18145161290322587, 0.39502164502164505, 1], [0.30241935483870974, 0.773809523809524, 1], [0.19153225806451613, 1.1931818181818183, 1], [0.4838709677419355, 1.6801948051948052, 1], [0.7560483870967742, 1.7343073593073597, 1], [1.2197580645161292, 1.9913419913419914, 1], [1.4112903225806452, 2.46482683982684, 1], [1.2096774193548387, 2.9383116883116887, 1], [1.189516129032258, 3.5470779220779223, 1], [1.3508064516129032, 4.034090909090909, 1], [2.006048387096774, 4.223484848484849, 1], [2.368951612903226, 3.8446969696969697, 1], [2.540322580645161, 3.5064935064935066, 1], [2.963709677419355, 3.4659090909090917, 1], [3.256048387096774, 3.979978354978355, 1], [3.377016129032258, 4.466991341991342, 1], [3.659274193548387, 4.76461038961039, 1], [4.203629032258064, 4.818722943722944, 1], [3.921370967741935, 4.76461038961039, 1], [3.548387096774193, 4.724025974025975, 1], [3.356854838709677, 4.169372294372295, 1], [2.157258064516129, 4.034090909090909, 1], [1.6431451612903227, 4.155844155844156, 1], [1.2802419354838708, 3.817640692640693, 1], [1.1995967741935483, 3.2629870129870135, 1], [1.3407258064516128, 2.7353896103896105, 1], [1.3205645161290323, 2.126623376623377, 1], [1.0080645161290323, 1.8425324675324677, 1], [0.32258064516129026, 1.4096320346320348, 1], [0.4233870967741935, 0.11093073593073588, 0], [0.5745967741935485, 0.43560606060606055, 0], [0.5443548387096775, 0.773809523809524, 0], [0.4536290322580645, 1.0714285714285716, 0], [0.4838709677419355, 1.314935064935065, 0], [0.967741935483871, 1.4772727272727275, 0], [1.4616935483870968, 1.693722943722944, 0], [1.7741935483870968, 2.113095238095238, 0], [1.7237903225806452, 2.4783549783549788, 0], [1.5423387096774193, 2.803030303030303, 0], [1.5020161290322582, 3.4659090909090917, 0], [1.5927419354838708, 3.6688311688311694, 0], [2.157258064516129, 3.6147186147186154, 0], [2.399193548387097, 3.276515151515152, 0], [1.9254032258064517, 3.168290043290044, 0], [2.006048387096774, 2.6812770562770565, 0], [2.520161290322581, 3.0465367965367967, 0], [3.235887096774193, 3.1141774891774894, 0], [3.4375, 3.6011904761904763, 0], [3.588709677419355, 4.196428571428572, 0], [3.830645161290322, 4.358766233766234, 0], [4.042338709677419, 4.534632034632035, 0], [4.475806451612903, 4.710497835497836, 0], [4.737903225806451, 4.737554112554113, 0], [0.7762096774193548, 1.4637445887445888, 0], [1.0786290322580645, 1.4502164502164505, 0], [2.842741935483871, 3.0059523809523814, 0], [1.8951612903225807, 3.6282467532467537, 0]]
tempData = trainingData
for j in range(4):
 for i in range(0, len(tempData)-1):
   trainingData.append([tempData[i][0]-(random.uniform(-0.1, 0.1)), tempData[i][1]-(random.uniform(-0.1, 0.1)), tempData[i][2]])
def train_and_save_model(i):
    print("__________________________________________________________")
    print("Model_A:", i)
    ANN = NeuralNetwork([4, 3, 5, 3, 1], 0.03, trainingData, i)
    ANN.train(2000, 100)
    ANN.calculate_f1_score(ANN.true_positives, ANN.false_positives, ANN.false_negatives)
    ANN.log_results("Legacy_Model_B_" + str(time.time()) + ".json")

if __name__ == "__main__":
    with multiprocessing.Pool() as pool:
        pool.map(train_and_save_model, range(0, 1))
