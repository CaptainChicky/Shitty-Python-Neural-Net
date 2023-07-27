import json
import numpy as np
from layer import Layer
from neural_network import NeuralNet
from training import Training


#####################
# Sample usage
#####################

# Load the neural net #

# Declare a neural network object
neural_net = NeuralNet()
# Load the neural network from the JSON file
neural_net.load("model_params.json")


# Train the neural net again #

# You may train the neural network again if you want to, perhaps even with new data
#
#   # Load data from the JSON file
#   with open("color_data.json", "r") as file:
#       data = json.load(file)
#   
#   # Extract the RGB values and Is_Red labels from the data
#   input_data = np.array(data["RBG_Values"])
#   target_data = np.array(data["Is_Red"])
#   
#   # Create a Training object with learning rates and clip values that you want
#   # You may have to adjust these values to be smaller, because the neural network is already trained into a local minimum
#   # So smaller learning rates and clip values may either get it stuck, or increase the cost function for some reason
#   # Experiment with it to get a balance because honestly it's just trial and error
#   training = Training(neural_net, learning_rate=0.00001, clip_value=4)
#   
#   # Train the neural network using your input data and target data for a specific number of epochs
#   num_epochs = 500
#   training.train(input_data, target_data, epochs=num_epochs)
#   
#   # Save the neural network
#   neural_net.save("model_params.json")


# Test the neural net #

# Create some sample input data
input_data = np.array([255, 0, 0]) # This is clearly red

# Perform forward propagation through the network
output_data = neural_net.forward_propagation(input_data)

# Percent confidence is just half of the distance between the two output neurons' values (x, y)
percent = (output_data[1] - output_data[0]) / 2

# If the distance is negative, then the color is red, otherwise it is not red
# This is specified in the definition of red python file
is_red = "is not red"
if percent < 0:
    is_red = "is red"
    percent *= -1

print("Verdict: the (r, g, b) color triple", is_red, "with", percent * 100, "% confidence.")