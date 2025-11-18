import json
import numpy as np
from neural_network import NeuralNet

############################################################################################################
# CHOOSE WHICH MODEL TO TEST:
# Uncomment ONE of the 5 configurations below
############################################################################################################

# CONFIGURATION 1: RGB Red Color Classification (Original)
# CONFIGURATION 2: XOR Problem
# CONFIGURATION 3: Sine Wave Classification
# CONFIGURATION 4: Checkerboard Pattern
# CONFIGURATION 5: Quadrant Classification (MULTI-CLASS - 4 outputs!)


############################################################################################################
# CONFIGURATION 1: RGB Red Color Classification
############################################################################################################
#model_file = "model_params.json"
#data_file = "color_data.json"
#input_key = "RGB_Values"
##output_key = "Is_Red"
#num_classes = 2
#class_names = ["Red", "Not Red"]
#print("=" * 70)
#print("TESTING: RGB Red Color Classification")
#print("=" * 70)


############################################################################################################
# # CONFIGURATION 2: XOR Problem
# ############################################################################################################
#model_file = "model_xor.json"
#data_file = "xor_data.json"
#input_key = "Input_Values"
#output_key = "Output_Values"
#num_classes = 2
#class_names = ["XOR=0", "XOR=1"]
#print("=" * 70)
#print("TESTING: XOR Problem")
#print("=" * 70)


############################################################################################################
# # CONFIGURATION 3: Sine Wave Classification
# ############################################################################################################
#model_file = "model_sine.json"
#data_file = "sine_data.json"
#input_key = "Input_Values"
#output_key = "Output_Values"
#num_classes = 2
#class_names = ["Below Sine", "Above Sine"]
#print("=" * 70)
#print("TESTING: Sine Wave Classification")
#print("=" * 70)


############################################################################################################
# # CONFIGURATION 4: Checkerboard Pattern
# ############################################################################################################
#model_file = "model_checkerboard.json"
#data_file = "checkerboard_data.json"
#input_key = "Input_Values"
#output_key = "Output_Values"
#num_classes = 2
#class_names = ["Black", "White"]
#print("=" * 70)
#print("TESTING: Checkerboard Pattern")
#print("=" * 70)


############################################################################################################
# # CONFIGURATION 5: Quadrant Classification (MULTI-CLASS!)
# ############################################################################################################
model_file = "model_quadrant.json"
data_file = "quadrant_data.json"
input_key = "Input_Values"
output_key = "Output_Values"
num_classes = 4  # MULTI-CLASS!
class_names = ["Q1 (x>0,y>0)", "Q2 (x<0,y>0)", "Q3 (x<0,y<0)", "Q4 (x>0,y<0)"]
print("=" * 70)
print("TESTING: Quadrant Classification (MULTI-CLASS)")
print("=" * 70)


############################################################################################################
# TESTING CODE (Works for both binary and multi-class)
############################################################################################################

# Load the neural net
neural_net = NeuralNet()
neural_net.load(model_file)

print(f"\nLoaded model from {model_file}")


############################################################################################################
# OPTIONAL: Continue Training a Loaded Model
############################################################################################################
# You may train the neural network again if you want to, perhaps even with new data
# This is useful for fine-tuning or training on additional data
#
# from training import Training
#
# # Load data from the JSON file
# with open(data_file, "r") as file:
#     data = json.load(file)
#
# # Extract the input values and output labels from the data
# input_data_train = np.array(data[input_key])
# target_data_train = np.array(data[output_key])
#
# # Create a Training object with learning rates and clip values that you want
# # You may have to adjust these values to be smaller, because the neural network is already trained into a local minimum
# # So smaller learning rates and clip values may either get it stuck, or increase the cost function for some reason
# # Experiment with it to get a balance because honestly it's just trial and error
# training = Training(neural_net, learning_rate=0.00001, clip_value=4)
#
# # Train the neural network using your input data and target data for a specific number of epochs
# num_epochs = 500
# training.train(input_data_train, target_data_train, epochs=num_epochs)
#
# # Save the neural network
# neural_net.save(model_file)


############################################################################################################
# Evaluate the Model on Test Data
############################################################################################################

# Load test data
with open(data_file, "r") as file:
    data = json.load(file)

input_data = np.array(data[input_key])
target_data = np.array(data[output_key])

print(f"Loaded {len(input_data)} test samples from {data_file}")
print()

# Make predictions
predictions = []
for i in range(len(input_data)):
    prediction = neural_net.forward_propagation(input_data[i])

    # Find which class has the highest output
    predicted_class = np.argmax(prediction)
    predictions.append(predicted_class)

# Convert targets to class indices
target_classes = []
for target in target_data:
    target_class = np.argmax(target)
    target_classes.append(target_class)

# Calculate metrics
num_correct = 0
confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

for i in range(len(target_classes)):
    if target_classes[i] == predictions[i]:
        num_correct += 1

    # Update confusion matrix: rows=actual, cols=predicted
    confusion_matrix[target_classes[i]][predictions[i]] += 1

accuracy = num_correct / len(target_classes) * 100

# Print results
print("=" * 70)
print("RESULTS")
print("=" * 70)
print(f"\nAccuracy: {accuracy:.2f}%")
print(f"Correct: {num_correct}/{len(target_classes)}")
print()

# Print confusion matrix
print("Confusion Matrix (rows=actual, cols=predicted):")
print()
header = "Actual \\ Pred  |" + "".join([f" {class_names[i]:^15}" for i in range(num_classes)])
print(header)
print("-" * len(header))
for i in range(num_classes):
    row = f"{class_names[i]:^15}|"
    for j in range(num_classes):
        row += f" {confusion_matrix[i][j]:^15}"
    print(row)
print()

# Calculate per-class metrics (Precision, Recall, F1)
############################################################################################################
# UNDERSTANDING THE METRICS:
############################################################################################################
# There are many different measurements of how well a neural net is doing for a given set of data
#
# ACCURACY:
# - Accuracy is only a decent metric if your dataset is roughly 50-50 between classes
# - We could have a neural net that spits a garbage answer of "not red" for every single input,
#   and it would still have very high accuracy if there aren't many red samples
# - This is why we also use precision and recall
#
# For binary classification, if we define the "positive" class (e.g., red) and "negative" class (e.g., not red):
#
# PRECISION (for a class):
# - Of all times the network predicted this class, how many were actually correct?
# - Formula: True Positives / (True Positives + False Positives)
# - Example: If network says "red" 100 times, and 95 were actually red, precision = 95%
#
# RECALL (for a class):
# - Of all actual samples of this class, how many did the network find?
# - Formula: True Positives / (True Positives + False Negatives)
# - Example: If there are 100 red samples, and network found 90 of them, recall = 90%
#
# F1-SCORE:
# - Harmonic mean of precision and recall
# - Formula: 2 * (Precision * Recall) / (Precision + Recall)
# - Balances precision and recall into a single score
#
# For multi-class (like quadrants), we calculate these metrics for EACH class separately
############################################################################################################

print("Per-Class Metrics:")
print("-" * 70)
print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("-" * 70)

for i in range(num_classes):
    # True Positives: correctly predicted as class i
    tp = confusion_matrix[i][i]

    # False Positives: incorrectly predicted as class i
    fp = sum(confusion_matrix[j][i] for j in range(num_classes) if j != i)

    # False Negatives: incorrectly predicted as not class i
    fn = sum(confusion_matrix[i][j] for j in range(num_classes) if j != i)

    # Calculate metrics
    if (tp + fp) > 0:
        precision = tp / (tp + fp) * 100
    else:
        precision = 0.0

    if (tp + fn) > 0:
        recall = tp / (tp + fn) * 100
    else:
        recall = 0.0

    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    print(f"{class_names[i]:<20} {precision:>6.2f}%      {recall:>6.2f}%      {f1:>6.2f}%")

print()
print("=" * 70)
print(f"Testing complete!")
print("=" * 70)