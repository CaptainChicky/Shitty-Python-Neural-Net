import json
import numpy as np
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from neural_network import NeuralNet
from training import Training

############################################################################################################
# CONFIGURATION SELECTOR - Change this number to select which model to continue training (1-7)
############################################################################################################
CONFIG_TO_RUN = 7

############################################################################################################
# CONFIGURATION DEFINITIONS
############################################################################################################

def get_configuration(config_num):
    """
    Returns the configuration for continuing training of the specified model.
    Lower learning rates are used for fine-tuning.
    """

    configs = {
        1: {
            'name': 'RGB Red Color Classification',
            'model_file': 'model_red.json',
            'data_file': 'color_data.json',
            'input_key': 'RGB_Values',
            'output_key': 'Is_Red',
            'learning_rate': 0.00005,  # Lower LR for fine-tuning
            'clip_value': 4,
            'num_epochs': 300,
            'num_samples': 900,
            'cost_function': 'mse' # we initially trained with mae, but we can switch cost functions! Now mse :)
            # like we can switch between mse ⟷ mae, but switching to or from a CE is trickly, refer to readme!!
        },

        2: {
            'name': 'XOR Problem',
            'model_file': 'model_xor.json',
            'data_file': 'xor_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'learning_rate': 0.0005,
            'clip_value': 4,
            'num_epochs': 300,
            'num_samples': 900,
            'cost_function': 'mse', # technically you can put `None` which defaults to `'mse'`. 
        },

        3: {
            'name': 'Sine Wave Classification',
            'model_file': 'model_sine.json',
            'data_file': 'sine_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'learning_rate': 0.00001,
            'clip_value': 4,
            'num_epochs': 500,
            'num_samples': 900,
            'cost_function': 'binary_crossentropy'
        },

        4: {
            'name': 'Checkerboard Pattern',
            'model_file': 'model_checkerboard.json',
            'data_file': 'checkerboard_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'learning_rate': 0.00005,
            'clip_value': 4,
            'num_epochs': 500,
            'num_samples': 900,
            'cost_function': 'mse',
        },

        5: {
            'name': 'Quadrant Classification (MULTI-CLASS)',
            'model_file': 'model_quadrant.json',
            'data_file': 'quadrant_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'learning_rate': 0.0001,
            'clip_value': 4,
            'num_epochs': 300,
            'num_samples': 900,
            'cost_function': 'mae',
        },

        6: {
            'name': 'House Price Regression (LINEAR OUTPUT)',
            'model_file': 'model_linear_regression.json',
            'data_file': 'linear_regression_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'learning_rate': 0.0001,
            'clip_value': 4,
            'num_epochs': 300,
            'num_samples': 900,
            'cost_function': 'mse',
        },

        7: {
            'name': 'Iris Flower Classification (SOFTMAX + CATEGORICAL CE)',
            'model_file': 'model_iris.json',
            'data_file': 'iris_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'learning_rate': 0.0005,
            'clip_value': 4,
            'num_epochs': 300,
            'num_samples': 900,
            'cost_function': 'categorical_crossentropy',
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
print(f"CONTINUE TRAINING: {config['name']}")
print("=" * 70)
print()

# Load the existing neural net
model_file = os.path.join(os.path.dirname(__file__), "models", config['model_file'])
neural_net = NeuralNet()
neural_net.load(model_file)
print(f"Loaded existing model from {config['model_file']}")

# Load training data
data_file = os.path.join(os.path.dirname(__file__), "data", config['data_file'])
with open(data_file, "r") as file:
    data = json.load(file)

input_data = np.array(data[config['input_key']])
target_data = np.array(data[config['output_key']])
print(f"Loaded {len(input_data)} training samples from {config['data_file']}")
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
print(f"Learning rate: {config['learning_rate']}")
print(f"Clip value: {config['clip_value']}")
print(f"Additional epochs: {config['num_epochs']}")
print(f"Cost function: {config['cost_function']}")
print()

# Create a Training object with LOWER learning rate for fine-tuning
# Use model_file as checkpoint to save best version during continue training
training = Training(neural_net, learning_rate=config['learning_rate'], clip_value=config['clip_value'], cost_function=config['cost_function'], checkpoint_path=model_file)

# Continue training
training.train(input_data, target_data, epochs=config['num_epochs'], samples_per_epoch=config['num_samples'])

# Best model already saved during training via checkpoint
print()
print("=" * 70)
print(f"Continue-training complete! Best model saved to {config['model_file']}")
print(f"Best cost achieved: {training.best_cost:.6f}")
print("=" * 70)
print()
print("TIP: Run main_load.py to see if performance improved")