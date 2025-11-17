import json
import math
import random

# The color red basically is defined as follows
# Create an RGB color triple (r, g, b) and graph it on a 3D coordinate system with each axis being one of r, g, or b
# The color red is defined as any point that is within 127 units (inclusive) of the point (255, 0, 0) in this 3D coordinate system
# Basically its like part of a sphere with radius 127 centered at (255, 0, 0), and anything in this sphere is considered red
def is_color_red(r, g, b):

    # Define the coordinates for the "definition" of the red point (255, 0, 0)
    red_point = (255, 0, 0)

    # Calculate the distance between the given color and the red point
    # 3D distance formula is as follows for two points (x1, y1, z1) and (x2, y2, z2):`
    # distance = sqrt((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2`
    distance = math.sqrt((r - red_point[0])**2 + (g - red_point[1])**2 + (b - red_point[2])**2)

    # Check if the distance is less than or equal to 127
    if distance <= 127:
        return True
    else:
        return False

# Generate what the neural net is supposed to output for a given color
def generate_output(isRed):
    # If the color is red, we want the output to be (1, -1)
    # If the color is not red, we want the output to be (-1, 1)
    if isRed:
        return (1, -1)
    else:
        return (-1, 1)
    

# Here is some more in depth explanation on what the output means: 
#   If we input a rbg value (r, b, g), we expect an output in the form (x, y) where x and y are between -1 and 1
#   The values are normalized between -1 and 1, so the absolute confidence that a color is red is (1, -1), and vice versa
#   We're basically looking for a difference of y - x = -2 when the color is red, and a difference of y - x = 2 when the color is not red
#   This is a wierd way to define red, because we can simply use one node in the neural net, but it is good practice to use two nodes as it is more generalizable
#   The reason we're not using values between 0 and 1 is because values between -1 and 1 have more flexibility and imo sigmoid just sucks as a function


# Generate training or testing data

# Function to generate random RGB triples (r, g, b) where r, g, and b are between 0 and 255
def generate_random_rgb():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Length of generation data
# You may modify this to generate more or less data
# Note that the more data you generate, the longer it will take to train the neural net
dataLength = 500


# Create a set to store unique RGB triples (we do not want duplicates)
unique_rgb_set = set()

# Keep generating random RGB triples until we have enough unique ones
while len(unique_rgb_set) < dataLength:
    unique_rgb_set.add(generate_random_rgb())


# Convert the set to a list
# This is the input we want the neural net to take in
data_entry_1 = list(unique_rgb_set)
    
# Create the second data entry by checking if each RGB triple is red or not
# This is the output we want the neural net to generate
data_entry_2 = []
for r, g, b in data_entry_1:
    is_red = is_color_red(r, g, b)
    data_entry_2.append(generate_output(is_red))


# Combine the data entries into a dictionary
data = {
    "RGB_Values": data_entry_1,
    "Is_Red": data_entry_2
}

# Save the data as a JSON file
with open("color_data.json", "w") as file:
    json.dump(data, file)




############################################################################################################
# Softmax is crazy lmao, this is just a theoretical implementation, haven't bothered to properly do it
############################################################################################################
# import numpy as np
# 
# class Layer:
#     # Other methods and attributes blah blah blah...
# 
#     def compute_propagation(self, input_data):
#         # Compute the net input for this layer
#         weighted_input = np.dot(self.weights, input_data) + self.biases
# 
#         # Apply the activation function based on the layer type
#         if self.layer_type == 'input':
#             output = input_data
#         elif self.layer_type == 'output':
#             # Apply Softmax activation for the output layer
#             output = self.softmax(weighted_input)
#         elif self.layer_type == 'hidden':
#             output = self.activation_func(weighted_input)
#         else:
#             raise ValueError("Invalid layer type.")
# 
#         return output
# 
#     def softmax(self, x):
#         # Softmax activation function
#         # The input is a 1D numpy array representing the weighted input for each node in the output layer
# 
#         # To avoid numerical instability, we subtract the maximum value from each element of x
#         # This doesn't affect the result, but it helps avoid very large exponentials that could lead to overflow
#         e_x = np.exp(x - np.max(x))
# 
#         # Calculate the softmax probabilities by dividing each element by the sum of all elements
#         softmax_probs = e_x / e_x.sum()
# 
#         return softmax_probs

# 