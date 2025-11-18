import json
import numpy as np
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from neural_network import NeuralNet
from training import Training

############################################################################################################
# CHOOSE WHICH MODEL TO CONTINUE TRAINING:
# Uncomment ONE of the 5 configurations below
############################################################################################################

# CONFIGURATION 1: RGB Red Color Classification
# CONFIGURATION 2: XOR Problem
# CONFIGURATION 3: Sine Wave Classification
# CONFIGURATION 4: Checkerboard Pattern
# CONFIGURATION 5: Quadrant Classification (MULTI-CLASS - 4 outputs!)


############################################################################################################
# CONFIGURATION 1: RGB Red Color Classification
############################################################################################################
model_file = os.path.join(os.path.dirname(__file__), "models", "model_red.json")
data_file = os.path.join(os.path.dirname(__file__), "data", "color_data.json")
input_key = "RGB_Values"
output_key = "Is_Red"
learning_rate = 0.000005  # Lower LR for fine-tuning
clip_value = 4
num_epochs = 100
print("=" * 70)
print("CONTINUE TRAINING: RGB Red Color Classification")
print("=" * 70)


############################################################################################################
# # CONFIGURATION 2: XOR Problem
# ############################################################################################################
# model_file = os.path.join(os.path.dirname(__file__), "models", "model_xor.json")
# data_file = os.path.join(os.path.dirname(__file__), "data", "xor_data.json")
# input_key = "Input_Values"
# output_key = "Output_Values"
# learning_rate = 0.00005  # Lower LR for fine-tuning
# clip_value = 4
# num_epochs = 100
# print("=" * 70)
# print("CONTINUE TRAINING: XOR Problem")
# print("=" * 70)


############################################################################################################
# # CONFIGURATION 3: Sine Wave Classification
# ############################################################################################################
# model_file = os.path.join(os.path.dirname(__file__), "models", "model_sine.json")
# data_file = os.path.join(os.path.dirname(__file__), "data", "sine_data.json")
# input_key = "Input_Values"
# output_key = "Output_Values"
# learning_rate = 0.000025  # Lower LR for fine-tuning
# clip_value = 4
# num_epochs = 200
# print("=" * 70)
# print("CONTINUE TRAINING: Sine Wave Classification")
# print("=" * 70)


############################################################################################################
# # CONFIGURATION 4: Checkerboard Pattern
# ############################################################################################################
# model_file = os.path.join(os.path.dirname(__file__), "models", "model_checkerboard.json")
# data_file = os.path.join(os.path.dirname(__file__), "data", "checkerboard_data.json")
# input_key = "Input_Values"
# output_key = "Output_Values"
# learning_rate = 0.00005  # Lower LR for fine-tuning
# clip_value = 4
# num_epochs = 200
# print("=" * 70)
# print("CONTINUE TRAINING: Checkerboard Pattern")
# print("=" * 70)


############################################################################################################
# # CONFIGURATION 5: Quadrant Classification (MULTI-CLASS!)
# ############################################################################################################
# model_file = os.path.join(os.path.dirname(__file__), "models", "model_quadrant.json")
# data_file = os.path.join(os.path.dirname(__file__), "data", "quadrant_data.json")
# input_key = "Input_Values"
# output_key = "Output_Values"
# learning_rate = 0.00005  # Lower LR for fine-tuning
# clip_value = 4
# num_epochs = 100
# print("=" * 70)
# print("CONTINUE TRAINING: Quadrant Classification (MULTI-CLASS)")
# print("=" * 70)


############################################################################################################
# CONTINUE TRAINING CODE
############################################################################################################

# Load the existing neural net
neural_net = NeuralNet()
neural_net.load(model_file)

print(f"\nLoaded existing model from {model_file}")

# Load training data
with open(data_file, "r") as file:
    data = json.load(file)

input_data = np.array(data[input_key])
target_data = np.array(data[output_key])

print(f"Loaded {len(input_data)} training samples from {data_file}")
print()

# Important notes about continuing training
print("=" * 70)
print("IMPORTANT: Continuing Training on Already-Trained Model")
print("=" * 70)
print("- Using LOWER learning rate than initial training")
print("- This is fine-tuning, not training from scratch")
print("- Cost may not decrease much if model is already well-trained")
print("- If cost INCREASES, learning rate is too high or model is overfitting")
print("=" * 70)
print()
print(f"Learning rate: {learning_rate}")
print(f"Clip value: {clip_value}")
print(f"Additional epochs: {num_epochs}")
print()

# Create a Training object with LOWER learning rate for fine-tuning
# Using a lower learning rate prevents the model from "forgetting" what it already learned
training = Training(neural_net, learning_rate=learning_rate, clip_value=clip_value)

# Continue training
training.train(input_data, target_data, epochs=num_epochs)

# Save the updated model (overwrites the old one)
neural_net.save(model_file)

print()
print("=" * 70)
print(f"Continue-training complete! Updated model saved to {model_file}")
print("=" * 70)
print()
print("TIP: Run main_load.py to see if performance improved")