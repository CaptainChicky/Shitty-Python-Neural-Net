import json
import numpy as np
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from layer import Layer
from neural_network import NeuralNet
from training import Training


############################################################################################################
# CHOOSE WHICH PROBLEM TO TRAIN:
# Uncomment ONE of the 5 configurations below
############################################################################################################

# CONFIGURATION 1: RGB Red Color Classification (Original)
# CONFIGURATION 2: XOR Problem (Classic)
# CONFIGURATION 3: Sine Wave Classification
# CONFIGURATION 4: Checkerboard Pattern
# CONFIGURATION 5: Quadrant Classification (MULTI-CLASS - 4 outputs!)


############################################################################################################
# CONFIGURATION 1: RGB Red Color Classification
############################################################################################################
# Task: Classify RGB colors as "red" or "not red"
# Input: 3 neurons (R, G, B values 0-255)
# Output: 2 neurons (red vs not-red)
# Architecture: 3 → 10 → 5 → 2

#neural_net = NeuralNet()
#input_layer = Layer(previousLayer_size=3, layer_size=3, layer_type='input')
#hidden_layer1 = Layer(previousLayer_size=3, layer_size=10, layer_type='hidden')
#hidden_layer2 = Layer(previousLayer_size=10, layer_size=5, layer_type='hidden')
#output_layer = Layer(previousLayer_size=5, layer_size=2, layer_type='output')
#neural_net.add_layer(input_layer)
#neural_net.add_layer(hidden_layer1)
#neural_net.add_layer(hidden_layer2)
#neural_net.add_layer(output_layer)
#
#data_file = os.path.join(os.path.dirname(__file__), "data", "color_data.json")
#input_key = "RGB_Values"
#output_key = "Is_Red"
#learning_rate = 0.00001
#num_epochs = 500
#save_file = os.path.join(os.path.dirname(__file__), "models", "model_red.json")
#print("=" * 70)
#print("TRAINING: RGB Red Color Classification")
#print("Architecture: 3 → 10 → 5 → 2")
#print("=" * 70)


############################################################################################################
# # CONFIGURATION 2: XOR Problem
# ############################################################################################################
# # Task: Learn XOR function (historically important - proves you need hidden layers!)
# # Input: 2 neurons (x1, x2 binary values with noise)
# # Output: 2 neurons (0 vs 1)
# # Architecture: 2 → 4 → 2 (smaller network for simpler problem)
#
#neural_net = NeuralNet()
#input_layer = Layer(previousLayer_size=2, layer_size=2, layer_type='input')
#hidden_layer1 = Layer(previousLayer_size=2, layer_size=4, layer_type='hidden')
#output_layer = Layer(previousLayer_size=4, layer_size=2, layer_type='output')
#neural_net.add_layer(input_layer)
#neural_net.add_layer(hidden_layer1)
#neural_net.add_layer(output_layer)
#
#data_file = os.path.join(os.path.dirname(__file__), "data", "xor_data.json")
#input_key = "Input_Values"
#output_key = "Output_Values"
#learning_rate = 0.001  # Higher learning rate for simpler problem
#num_epochs = 500
#save_file = os.path.join(os.path.dirname(__file__), "models", "model_xor.json")
#print("=" * 70)
#print("TRAINING: XOR Problem (Classic Neural Network Test)")
#print("Architecture: 2 → 4 → 2")
#print("=" * 70)


############################################################################################################
# # CONFIGURATION 3: Sine Wave Classification
# ############################################################################################################
# # Task: Classify points as above or below y = sin(x)
# # Input: 2 neurons (x, y coordinates)
# # Output: 2 neurons (above vs below)
# # Architecture: 2 → 12 → 8 → 2 (more neurons to learn periodic pattern)
#
#neural_net = NeuralNet()
#input_layer = Layer(previousLayer_size=2, layer_size=2, layer_type='input')
#hidden_layer1 = Layer(previousLayer_size=2, layer_size=12, layer_type='hidden')
#hidden_layer2 = Layer(previousLayer_size=12, layer_size=8, layer_type='hidden')
#output_layer = Layer(previousLayer_size=8, layer_size=2, layer_type='output')
#neural_net.add_layer(input_layer)
#neural_net.add_layer(hidden_layer1)
#neural_net.add_layer(hidden_layer2)
#neural_net.add_layer(output_layer)
#
#data_file = os.path.join(os.path.dirname(__file__), "data", "sine_data.json")
#input_key = "Input_Values"
#output_key = "Output_Values"
#learning_rate = 0.0001  # Lower learning rate for smooth pattern
#num_epochs = 5000  # More epochs for periodic pattern
#save_file = os.path.join(os.path.dirname(__file__), "models", "model_sine.json")
#print("=" * 70)
#print("TRAINING: Sine Wave Classification")
#print("Architecture: 2 → 12 → 8 → 2")
#print("Task: Points above/below y = sin(x)")
#print("=" * 70)


############################################################################################################
# # CONFIGURATION 4: Checkerboard Pattern
# ############################################################################################################
# # Task: Classify grid squares as black or white (like a chess board)
# # Input: 2 neurons (x, y coordinates)
# # Output: 2 neurons (black vs white)
# # Architecture: 2 → 16 → 8 → 2 (many neurons for multiple decision boundaries)
#
#neural_net = NeuralNet()
#input_layer = Layer(previousLayer_size=2, layer_size=2, layer_type='input')
#hidden_layer1 = Layer(previousLayer_size=2, layer_size=16, layer_type='hidden')
#hidden_layer2 = Layer(previousLayer_size=16, layer_size=8, layer_type='hidden')
#hidden_layer3 = Layer(previousLayer_size=8, layer_size=8, layer_type='hidden')
#output_layer = Layer(previousLayer_size=8, layer_size=2, layer_type='output')
#neural_net.add_layer(input_layer)
#neural_net.add_layer(hidden_layer1)
#neural_net.add_layer(hidden_layer2)
#neural_net.add_layer(output_layer)
#
#data_file = os.path.join(os.path.dirname(__file__), "data", "checkerboard_data.json")
#input_key = "Input_Values"
#output_key = "Output_Values"
#learning_rate = 0.0005
#num_epochs = 5000
#save_file = os.path.join(os.path.dirname(__file__), "models", "model_checkerboard.json")
#print("=" * 70)
#print("TRAINING: Checkerboard Pattern")
#print("Architecture: 2 → 16 → 8 → 8 → 2")
#print("Task: Classify checkerboard squares")
#print("=" * 70)


############################################################################################################
# # CONFIGURATION 5: Quadrant Classification (MULTI-CLASS!)
# ############################################################################################################
# # Task: Classify which quadrant a point is in (4 classes!)
# # Input: 2 neurons (x, y coordinates)
# # Output: 4 neurons (Q1, Q2, Q3, Q4) - ONE-HOT ENCODING
# # Architecture: 2 → 8 → 6 → 4 (NOTE: 4 output neurons for 4 classes!)
#
neural_net = NeuralNet()
input_layer = Layer(previousLayer_size=2, layer_size=2, layer_type='input')
hidden_layer1 = Layer(previousLayer_size=2, layer_size=10, layer_type='hidden')
hidden_layer2 = Layer(previousLayer_size=10, layer_size=8, layer_type='hidden')
output_layer = Layer(previousLayer_size=8, layer_size=4, layer_type='output')  # 4 outputs!
neural_net.add_layer(input_layer)
neural_net.add_layer(hidden_layer1)
neural_net.add_layer(hidden_layer2)
neural_net.add_layer(output_layer)

data_file = os.path.join(os.path.dirname(__file__), "data", "quadrant_data.json")
input_key = "Input_Values"
output_key = "Output_Values"
learning_rate = 0.0005
num_epochs = 1000
save_file = os.path.join(os.path.dirname(__file__), "models", "model_quadrant.json")
print("=" * 70)
print("TRAINING: Quadrant Classification (MULTI-CLASS)")
print("Architecture: 2 → 8 → 6 → 4")
print("Task: Classify points into 4 quadrants")
print("Output: One-hot [Q1, Q2, Q3, Q4]")
print("=" * 70)


############################################################################################################
# TRAINING CODE (Same for all configurations)
############################################################################################################

# Load training data
with open(data_file, "r") as file:
    data = json.load(file)

input_data = np.array(data[input_key])
target_data = np.array(data[output_key])

print(f"\nLoaded {len(input_data)} training samples from {data_file}")
print(f"Learning rate: {learning_rate}")
print(f"Epochs: {num_epochs}")
print()

# Create a Training object
training = Training(neural_net, learning_rate=learning_rate, clip_value=5)

# Train the neural network
training.train(input_data, target_data, epochs=num_epochs)

# Save the neural net
neural_net.save(save_file)

print()
print("=" * 70)
print(f"Training complete! Model saved to {save_file}")
print("=" * 70)