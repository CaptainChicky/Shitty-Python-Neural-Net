import json
import numpy as np
from layer import Layer
from neural_network import NeuralNet
from training import Training

# Sample usage
neural_net = NeuralNet()

# Input layer "previousLayer size" parameter should always be its own size
input_layer = Layer(previousLayer_size=3, layer_size=3, layer_type='input')
hidden_layer1 = Layer(previousLayer_size=3, layer_size=10, layer_type='hidden')
hidden_layer2 = Layer(previousLayer_size=10, layer_size=5, layer_type='hidden')
output_layer = Layer(previousLayer_size=5, layer_size=2, layer_type='output')
neural_net.add_layer(input_layer)
neural_net.add_layer(hidden_layer1)
neural_net.add_layer(hidden_layer2)
neural_net.add_layer(output_layer)

# This is a neural net with 3 input neurons, 2 hidden layers with 10 and 5 neurons respectively, and 2 output neurons


# Load data from the JSON file
with open("color_data.json", "r") as file:
    data = json.load(file)

# Extract the RGB values and Is_Red labels from the data
input_data = np.array(data["RBG_Values"])
target_data = np.array(data["Is_Red"])

# Create a Training object
training = Training(neural_net, learning_rate=0.001, clip_value=5)

# Train the neural network using your input data and target data for a specific number of epochs
num_epochs = 500
training.train(input_data, target_data, epochs=num_epochs)



neural_net.save("model_params.json")