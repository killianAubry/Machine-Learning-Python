# Parallel Neural Network Model Training with Multiprocessing
## Overview
This file (Legacy_Model.py) exemplifies parallel model training for neural networks using the multiprocessing module in Python. The script is designed to create and train multiple neural network models concurrently, allowing for experimentation with different architectures, learning rates, and other parameters. The code structure is adapted from the provided neural network implementation.

## Features
* Multiprocessing: Leverages parallel processing to train multiple neural network models simultaneously, optimizing training time.

* Configurable Parameters: Customize neural network architectures, learning rates, and other parameters easily using the NeuralNetwork class.

## Example Usage
*Example Configuration
```
#sets the architecture for a neural network to 4 inputs nodes, 3 hidden layers with 3, 5, and 3 nodes respectivley and 1 output node
architecture = [4, 3, 5, 3, 1]

#adjusts the learning rate
learning_rate = 0.03

# Example: Change the number of training epochs
epochs = 1500

# Creates a new neural network with specified architecture, learning_rate, training_Data, and seed
 ANN = NeuralNetwork(architecture, learning_rate, training_Data, seed)

#trains the neural network for 2000 epochs and graphs the predictions every 100 epochs, if set to False it will not graph 
 ANN.train(2000, 100)
```

Contributions to enhance the programs are welcome. Feel free to open issues, submit pull requests, or provide feedback. Your input will be valuable for refining the script and making it more versatile.

License
This script is licensed under the MIT License - see the LICENSE.md file for details.
