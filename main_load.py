import json
import numpy as np
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from neural_network import NeuralNet

############################################################################################################
# CONFIGURATION SELECTOR - Change this number to select which model to test (1-7)
############################################################################################################
CONFIG_TO_RUN = 4

############################################################################################################
# CONFIGURATION DEFINITIONS
############################################################################################################

def get_configuration(config_num):
    """Returns the configuration for loading and testing the specified model."""

    configs = {
        1: {
            'name': 'RGB Red Color Classification',
            'model_file': 'model_red.json',
            'data_file': 'color_data.json',
            'input_key': 'RGB_Values',
            'output_key': 'Is_Red',
            'num_classes': 2,
            'class_names': ['Red', 'Not Red']
        },

        2: {
            'name': 'XOR Problem',
            'model_file': 'model_xor.json',
            'data_file': 'xor_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'num_classes': 2,
            'class_names': ['XOR=0', 'XOR=1']
        },

        3: {
            'name': 'Sine Wave Classification',
            'model_file': 'model_sine.json',
            'data_file': 'sine_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'num_classes': 2,
            'class_names': ['Below Sine', 'Above Sine']
        },

        4: {
            'name': 'Checkerboard Pattern',
            'model_file': 'model_checkerboard.json',
            'data_file': 'checkerboard_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'num_classes': 2,
            'class_names': ['Black', 'White']
        },

        5: {
            'name': 'Quadrant Classification (MULTI-CLASS)',
            'model_file': 'model_quadrant.json',
            'data_file': 'quadrant_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'num_classes': 4,
            'class_names': ['Q1 (x>0,y>0)', 'Q2 (x<0,y>0)', 'Q3 (x<0,y<0)', 'Q4 (x>0,y<0)']
        },

        6: {
            'name': 'House Price Regression (LINEAR OUTPUT)',
            'model_file': 'model_linear_regression.json',
            'data_file': 'linear_regression_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'num_classes': 1,  # Regression - single output
            'class_names': ['Price']
        },

        7: {
            'name': 'Iris Flower Classification (SOFTMAX + CATEGORICAL CE)',
            'model_file': 'model_iris.json',
            'data_file': 'iris_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'num_classes': 3,
            'class_names': ['Setosa', 'Versicolor', 'Virginica']
        },
    }

    if config_num not in configs:
        raise ValueError(f"Invalid configuration number: {config_num}. Must be 1-7.")

    return configs[config_num]


############################################################################################################
# MAIN EXECUTION
############################################################################################################

# Get the selected configuration
config = get_configuration(CONFIG_TO_RUN)

# Print configuration info
print("=" * 70)
print(f"TESTING: {config['name']}")
print("=" * 70)
print()

# Load the neural net
model_file = os.path.join(os.path.dirname(__file__), "models", config['model_file'])
neural_net = NeuralNet()
neural_net.load(model_file)
print(f"Loaded model from {config['model_file']}")

############################################################################################################
# Evaluate the Model on Test Data
############################################################################################################

# Load test data
data_file = os.path.join(os.path.dirname(__file__), "data", config['data_file'])
with open(data_file, "r") as file:
    data = json.load(file)

input_data = np.array(data[config['input_key']])
target_data = np.array(data[config['output_key']])

print(f"Loaded {len(input_data)} test samples from {config['data_file']}")
print()

# Check if this is regression (single continuous output) or classification (multiple classes)
is_regression = (config['num_classes'] == 1)

if is_regression:
    ############################################################################################################
    # REGRESSION EVALUATION
    ############################################################################################################
    print("=" * 70)
    print("REGRESSION EVALUATION")
    print("=" * 70)
    print()

    # Make predictions
    predictions = []
    for i in range(len(input_data)):
        prediction = neural_net.forward_propagation(input_data[i])
        # For regression, the output is the predicted value itself (not argmax)
        predictions.append(prediction[0])  # Single output value

    # Convert to numpy arrays for easier calculation
    predictions = np.array(predictions)
    targets = np.array([t[0] for t in target_data])  # Extract single target values

    # Calculate regression metrics
    errors = predictions - targets
    absolute_errors = np.abs(errors)
    squared_errors = errors ** 2

    # Mean Absolute Error (MAE): Average of absolute errors
    mae = np.mean(absolute_errors)

    # Mean Squared Error (MSE): Average of squared errors
    mse = np.mean(squared_errors)

    # Root Mean Squared Error (RMSE): Square root of MSE (same units as target)
    rmse = np.sqrt(mse)

    # R² Score (Coefficient of Determination): Proportion of variance explained
    # R² = 1 - (SS_res / SS_tot)
    # SS_res = sum of squared residuals (prediction errors)
    # SS_tot = total sum of squares (variance in targets)
    ss_res = np.sum(squared_errors)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Print results
    print("RESULTS:")
    print("-" * 70)
    print(f"Mean Absolute Error (MAE):       {mae:.4f}")
    print(f"Mean Squared Error (MSE):        {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE):  {rmse:.4f}")
    print(f"R² Score:                        {r2_score:.4f}")
    print()
    print("INTERPRETATION:")
    print(f"  - On average, predictions are off by {mae:.2f} units (MAE)")
    print(f"  - RMSE of {rmse:.2f} penalizes large errors more than MAE")
    print(f"  - R² of {r2_score:.4f} means {r2_score*100:.2f}% of variance is explained")
    if r2_score > 0.9:
        print("  - R² > 0.9: Excellent fit!")
    elif r2_score > 0.7:
        print("  - R² > 0.7: Good fit")
    elif r2_score > 0.5:
        print("  - R² > 0.5: Moderate fit")
    else:
        print("  - R² < 0.5: Poor fit - model may need more training")
    print()

    # Show some example predictions
    print("SAMPLE PREDICTIONS:")
    print("-" * 70)
    print(f"{'Target':<12} {'Predicted':<12} {'Error':<12} {'Abs Error':<12}")
    print("-" * 70)
    num_samples_to_show = min(10, len(predictions))
    for i in range(num_samples_to_show):
        print(f"{targets[i]:<12.2f} {predictions[i]:<12.2f} {errors[i]:<12.2f} {absolute_errors[i]:<12.2f}")
    print()

else:
    ############################################################################################################
    # CLASSIFICATION EVALUATION
    ############################################################################################################

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
    confusion_matrix = [[0 for _ in range(config['num_classes'])] for _ in range(config['num_classes'])]

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
    header = "Actual \\ Pred  |" + "".join([f" {config['class_names'][i]:^15}" for i in range(config['num_classes'])])
    print(header)
    print("-" * len(header))
    for i in range(config['num_classes']):
        row = f"{config['class_names'][i]:^15}|"
        for j in range(config['num_classes']):
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

    for i in range(config['num_classes']):
        # True Positives: correctly predicted as class i
        tp = confusion_matrix[i][i]

        # False Positives: incorrectly predicted as class i
        fp = sum(confusion_matrix[j][i] for j in range(config['num_classes']) if j != i)

        # False Negatives: incorrectly predicted as not class i
        fn = sum(confusion_matrix[i][j] for j in range(config['num_classes']) if j != i)

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

        print(f"{config['class_names'][i]:<20} {precision:>6.2f}%      {recall:>6.2f}%      {f1:>6.2f}%")

print()
print("=" * 70)
print(f"Testing complete!")
print("=" * 70)