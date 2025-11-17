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


############################################################################################################
# IMPORTANT: CLASS BALANCE IN TRAINING DATA
############################################################################################################
# When training a neural network for classification, it's critical to have BALANCED data
# (equal numbers of each class). Here's why:
#
# PROBLEM: If you generate 500 random RGB colors, you'll get roughly:
#   - 95% "not red" colors (473 samples)
#   - 5% "red" colors (27 samples)
#
# This is because the "red" definition (sphere radius 127 centered at (255,0,0)) only covers
# a small portion of the entire RGB color space (256^3 = 16.7 million possible colors).
#
# WHY THIS BREAKS TRAINING:
# The neural network learns to minimize cost (error). With 95% "not red" samples, it discovers
# a lazy strategy: "Always predict 'not red' → 95% accuracy!"
#
# This gives:
#   - High accuracy (95%)
#   - Zero precision (never correctly predicts red)
#   - Zero recall (never detects red at all)
#   - The network literally never outputs "red" for any input
#
# SOLUTION: Generate BALANCED data with 50% red and 50% not-red samples
# This forces the network to actually learn the pattern instead of exploiting class imbalance.
############################################################################################################

# Number of samples per class (red and not-red)
# Total dataset will be 2x this number
samples_per_class = 250  # 250 red + 250 not-red = 500 total


# Separate sets for red and not-red samples to ensure balance
red_samples = set()
not_red_samples = set()

print("Generating balanced dataset...")
print(f"Target: {samples_per_class} red samples, {samples_per_class} not-red samples")

# Keep generating until we have enough of EACH class
# This ensures 50-50 balance instead of the natural ~5-95 imbalance
attempts = 0
max_attempts = 100000  # Safety limit to prevent infinite loop

while (len(red_samples) < samples_per_class or len(not_red_samples) < samples_per_class) and attempts < max_attempts:
    rgb = generate_random_rgb()
    attempts += 1

    # Only add to the appropriate set if that set isn't full yet
    if is_color_red(*rgb) and len(red_samples) < samples_per_class:
        red_samples.add(rgb)
    elif not is_color_red(*rgb) and len(not_red_samples) < samples_per_class:
        not_red_samples.add(rgb)

    # Progress update every 10,000 attempts
    if attempts % 10000 == 0:
        print(f"  Attempt {attempts}: {len(red_samples)} red, {len(not_red_samples)} not-red")

print(f"\nGeneration complete after {attempts} attempts:")
print(f"  Red samples: {len(red_samples)}")
print(f"  Not-red samples: {len(not_red_samples)}")

# Combine both sets and shuffle to mix red and not-red samples randomly
# (important so the network doesn't see all reds first, then all not-reds)
all_samples = list(red_samples) + list(not_red_samples)
random.shuffle(all_samples)

# Create input data (RGB values) and output data (red or not-red labels)
data_entry_1 = all_samples
data_entry_2 = []
for r, g, b in data_entry_1:
    is_red = is_color_red(r, g, b)
    data_entry_2.append(generate_output(is_red))

# Verify the balance (should be 50-50)
num_red = sum(1 for output in data_entry_2 if output == (1, -1))
num_not_red = sum(1 for output in data_entry_2 if output == (-1, 1))
print(f"\nFinal dataset composition:")
print(f"  Red: {num_red} ({num_red/len(data_entry_2)*100:.1f}%)")
print(f"  Not red: {num_not_red} ({num_not_red/len(data_entry_2)*100:.1f}%)")

# Combine the data entries into a dictionary
data = {
    "RGB_Values": data_entry_1,
    "Is_Red": data_entry_2
}

# Save the data as a JSON file
with open("color_data.json", "w") as file:
    json.dump(data, file)

print("\nSaved to color_data.json")




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