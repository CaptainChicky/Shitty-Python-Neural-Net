import json
import numpy as np
from layer import Layer
from neural_network import NeuralNet
from training import Training



#####################
# Sample usage
#####################

# Make the neural net #

# Declare a neural network object
neural_net = NeuralNet()

# Declare the layers and add them to the neural network
# This is a neural net with 3 input neurons, 2 hidden layers with 10 and 5 neurons respectively, and 2 output neurons
# Input layer "previousLayer size" parameter should always be its own size
input_layer = Layer(previousLayer_size=3, layer_size=3, layer_type='input')
hidden_layer1 = Layer(previousLayer_size=3, layer_size=10, layer_type='hidden')
hidden_layer2 = Layer(previousLayer_size=10, layer_size=5, layer_type='hidden')
output_layer = Layer(previousLayer_size=5, layer_size=2, layer_type='output')
neural_net.add_layer(input_layer)
neural_net.add_layer(hidden_layer1)
neural_net.add_layer(hidden_layer2)
neural_net.add_layer(output_layer)



# Train the neural net #

# Load training data from the JSON file
with open("color_data.json", "r") as file:
    data = json.load(file)

# Extract the RGB values and Is_Red labels from the data
input_data = np.array(data["RBG_Values"])
target_data = np.array(data["Is_Red"])

# Create a Training object
# Clip value should be around 1 to 5, but you can set it as high as you want (if you get exploding gradients, then lmfao)
# Learning rate should be around 0.00001 to 0.0001
# It's hard to strike the right training value first try, so experiment with it, because te larger the training value, the more likely you 
# may not be able to settle down on a minimum and bounce over the place, meaning that the cost actually increases as you train
# If you set the learning rate too low, it will take a very, very long time to train
training = Training(neural_net, learning_rate=0.00001, clip_value=5)

# Train the neural network using your input data and target data for a specific number of epochs
# The larger the input dataset, and the more epochs, the longer it will take to train (an epoch is a single pass through the entire dataset)
# Don't make the input dataset to be too large, or else it will take quite a while to train for a single epoch
num_epochs = 500
training.train(input_data, target_data, epochs=num_epochs)



# Save the neural net #

# Everything is stored in a json file
neural_net.save("model_params.json")