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

# This is for one sample only!
# With this method, it outputs simply the accuracy of the neural network with a confidence level as well
#
# # Create some sample input data
# input_data = np.array([0, 12, 13]) # This is clearly red, but you can change it to whatever you want
# 
# # Perform forward propagation through the network
# output_data = neural_net.forward_propagation(input_data)
# 
# # Percent confidence is just half of the distance between the two output neurons' values (x, y)
# percent = (output_data[1] - output_data[0]) / 2
# 
# # If the distance is negative, then the color is red, otherwise it is not red
# # This is specified in the definition of red python file
# is_red = "is not red"
# if percent < 0:
#     is_red = "is red"
#     percent *= -1
# 
# print("Verdict: the (r, g, b) color triple", is_red, "with {:.16f}% confidence.".format(percent * 100))


# Accuracy measurements over a dataset 

# There are many different measurements of how well a neural net is doing for a given set of data
# However, accuracy is only a decent metric if your dataset is sploy 50-50, which our dataset is not
# We could have a neural net that spits a garbage answer of "not red" for every single input, and it would still have very high accuracy
# because there aren't a lot of colors that are considered "red" based on the way we defined it
# Instead, we use two different metrics, precision and recall, to measure how well the neural net is doing
# If we define the "positive" class as the minority (red) and the "negative" class as the majority (not red)
# Then precision is the propotion of the positive predictions that are correct, and recall is the propotion of the positive cases that were predicted correctly
# This would be a much better metric for our kind of binary classification case

# Load the test data from color_data.json (this should be a newly generated set of data)
with open("color_data.json", "r") as file:
    data = json.load(file)

input_data = np.array(data["RBG_Values"])
target_data = np.array(data["Is_Red"])

# Load the test data from color_data.json
with open("color_data.json", "r") as file:
    data = json.load(file)

input_data = np.array(data["RBG_Values"])
target_data = np.array(data["Is_Red"])

# Test the neural net
tempPredictionStore = np.empty((0, 2))  # Initialize with two columns for positive and negative predictions

for i in range(len(input_data)):
    prediction = neural_net.forward_propagation(input_data[i])
    if prediction[1] > prediction[0]:
        tempPredictionStore = np.append(tempPredictionStore, np.array([[-1, 1]]), axis=0)
    else:
        tempPredictionStore = np.append(tempPredictionStore, np.array([[1, -1]]), axis=0)

Num_correct = 0
True_positives = 0
False_positives = 0
False_negatives = 0

for i in range(len(target_data)):
    if np.array_equal(target_data[i], tempPredictionStore[i]):
        Num_correct += 1
        if np.array_equal(target_data[i], [-1, 1]):  # Check if it's a true positive
            True_positives += 1
    else:
        if np.array_equal(target_data[i], [-1, 1]):  # Check if it's a false positive
            False_positives += 1
        else:  # Check if it's a false negative
            False_negatives += 1

accuracy = Num_correct / len(target_data) * 100
precision = True_positives / (True_positives + False_positives) * 100
recall = True_positives / (True_positives + False_negatives) * 100

print("Accuracy: {:.2f}%".format(accuracy))
print("Precision: {:.2f}%".format(precision))
print("Recall: {:.2f}%".format(recall))
