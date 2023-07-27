import json
import numpy as np
from layer import Layer
from neural_network import NeuralNet
from training import Training

# Sample usage
neural_net = NeuralNet()
neural_net.load("model_params.json")



## Load data from the JSON file
#with open("color_data.json", "r") as file:
#    data = json.load(file)
#
## Extract the RGB values and Is_Red labels from the data
#input_data = np.array(data["RBG_Values"])
#target_data = np.array(data["Is_Red"])
#
## Create a Training object
#training = Training(neural_net, learning_rate=0.00001, clip_value=4)
#
## Train the neural network using your input data and target data for a specific number of epochs
#num_epochs = 500
#training.train(input_data, target_data, epochs=num_epochs)
#
#
#
#neural_net.save("model_params.json")



# Create some sample input data
input_data = np.array([230, 115, 139])

# Perform forward propagation through the network
output_data = neural_net.forward_propagation(input_data)

percent = (output_data[1] - output_data[0]) / 2

is_red = "is not red"
if percent < 0:
    is_red = "is red"
    percent *= -1

print("Verdict: the (r, g, b) color triple", is_red, "with", percent * 100, "% confidence.")
