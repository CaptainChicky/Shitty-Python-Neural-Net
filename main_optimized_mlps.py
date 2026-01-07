import json
import numpy as np
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from layer import Layer
from neural_network import NeuralNet
from training import Training
from activation_functions import ActivationFunction

############################################################################################################
# OPTIMIZED CONFIGURATION SELECTOR - Change this number to select which model to train (1-8)
############################################################################################################
CONFIG_TO_RUN = 3

############################################################################################################
# OPTIMIZED CONFIGURATION DEFINITIONS
############################################################################################################

def get_configuration(config_num):
    """
    Returns optimized configuration for maximum performance.
    These configs prioritize accuracy over feature showcase.
    """

    configs = {
        ############################################################################################################
        # CONFIG 1: RGB Red - BINARY CLASSIFICATION OPTIMIZED
        ############################################################################################################
        1: {
            'name': 'RGB Red Color Classification (OPTIMIZED)',
            'description': 'Classify RGB colors as "red" or "not red"',
            'architecture': '3 → 20 → 16 → 1',
            'details': 'Swish throughout (smooth for curved boundary) + tanh output',
            'layers': lambda: [
                Layer(3, 3, 'input'),
                Layer(3, 20, 'hidden',
                      activation_func=ActivationFunction.swish,
                      activation_params={'alpha': 1.0},
                      weight_init='normal',
                      weight_init_params={'std': 0.002},
                      bias_init='constant',
                      bias_init_params={'value': -1.0}),  # Shift to center around 0
                Layer(20, 16, 'hidden',
                      activation_func=ActivationFunction.swish,
                      activation_params={'alpha': 1.0},
                      weight_init='normal',
                      weight_init_params={'std': 0.001},
                      bias_init='zeros'),
                Layer(16, 1, 'output',
                      activation_func=ActivationFunction.tanh)
            ],
            'data_file': 'color_data.json',
            'input_key': 'RGB_Values',
            'output_key': 'Is_Red',
            'learning_rate': 0.002,
            'num_epochs': 1500,
            'num_samples': 1500,
            'cost_function': 'mse',
            'save_file': 'model_optimized_red.json'
        },
        ############################################################################################################
        # CONFIG 2: XOR - CLASSIC PROBLEM OPTIMIZED
        ############################################################################################################
        2: {
            'name': 'XOR Problem (OPTIMIZED)',
            'description': 'Learn XOR function with maximum efficiency',
            'architecture': '2 → 8 → 6 → 1',
            'details': 'Leaky ReLU throughout + He init + Sigmoid output + MSE',
            'layers': lambda: [
                Layer(2, 2, 'input'),
                Layer(2, 8, 'hidden',
                      activation_func=ActivationFunction.leaky_relu,
                      activation_params={'alpha': 0.01},
                      weight_init='he',
                      bias_init='zeros'),
                Layer(8, 6, 'hidden',
                      activation_func=ActivationFunction.leaky_relu,
                      activation_params={'alpha': 0.01},
                      weight_init='he',
                      bias_init='zeros'),
                Layer(6, 1, 'output',
                      activation_func=ActivationFunction.sigmoid)
            ],
            'data_file': 'xor_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'learning_rate': 0.001,
            'num_epochs': 1000,
            'num_samples': 900,
            'cost_function': 'binary_crossentropy',
            'save_file': 'model_optimized_xor.json'
        },

        ############################################################################################################
        # CONFIG 3: SINE WAVE - SMOOTH BOUNDARY OPTIMIZED
        ############################################################################################################
        3: {
            'name': 'Sine Wave Classification (OPTIMIZED)',
            'description': 'Classify points as above or below y = sin(x)',
            'architecture': '2 → 24 → 32 → 24 → 1',
            'details': 'Leaky ReLU throughout + He init + Sigmoid output + Binary CE',
            'layers': lambda: [
                Layer(2, 2, 'input'),
                Layer(2, 24, 'hidden',
                      activation_func=ActivationFunction.swish,
                      activation_params={'alpha': 1.0},
                      weight_init='he',
                      bias_init='zeros'),
                Layer(24, 32, 'hidden',
                      activation_func=ActivationFunction.swish,
                      activation_params={'alpha': 1.0},
                      weight_init='he',
                      bias_init='zeros'),
                Layer(32, 24, 'hidden',
                      activation_func=ActivationFunction.swish,
                      activation_params={'alpha': 1.0},
                      weight_init='he',
                      bias_init='zeros'),
                Layer(24, 1, 'output',
                      activation_func=ActivationFunction.sigmoid)
            ],
            'data_file': 'sine_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'learning_rate': 0.01,
            'num_epochs': 3000,
            'num_samples': 1200,
            'cost_function': 'binary_crossentropy',
            'save_file': 'model_optimized_sine.json'
        },

        ############################################################################################################
        # CONFIG 4: CHECKERBOARD - COMPLEX PATTERN OPTIMIZED
        ############################################################################################################
        4: {
            'name': 'Checkerboard Pattern (OPTIMIZED)',
            'description': 'Classify grid squares as black or white',
            'architecture': '2 → 40 → 50 → 40 → 1',
            'details': 'Leaky ReLU throughout + He init + Tanh output + MSE',
            'layers': lambda: [
                Layer(2, 2, 'input'),
                Layer(2, 40, 'hidden',
                      activation_func=ActivationFunction.leaky_relu,
                      activation_params={'alpha': 0.01},
                      weight_init='he',
                      bias_init='zeros'),
                Layer(40, 50, 'hidden',
                      activation_func=ActivationFunction.leaky_relu,
                      activation_params={'alpha': 0.01},
                      weight_init='he',
                      bias_init='zeros'),
                Layer(50, 40, 'hidden',
                      activation_func=ActivationFunction.leaky_relu,
                      activation_params={'alpha': 0.01},
                      weight_init='he',
                      bias_init='zeros'),
                Layer(40, 1, 'output',
                      activation_func=ActivationFunction.tanh)
            ],
            'data_file': 'checkerboard_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'learning_rate': 0.001,
            'num_epochs': 5000,
            'num_samples': 900,
            'cost_function': 'mse',
            'save_file': 'model_optimized_checkerboard.json'
        },

        ############################################################################################################
        # CONFIG 5: QUADRANT - MULTI-CLASS OPTIMIZED (4 classes)
        ############################################################################################################
        5: {
            'name': 'Quadrant Classification (OPTIMIZED)',
            'description': 'Classify which quadrant a point is in (4 classes)',
            'architecture': '2 → 30 → 30 → 4',
            'details': 'Leaky ReLU throughout + He init + Tanh output + MSE',
            'layers': lambda: [
                Layer(2, 2, 'input'),
                Layer(2, 30, 'hidden',
                      activation_func=ActivationFunction.leaky_relu,
                      activation_params={'alpha': 0.01},
                      weight_init='he',
                      bias_init='zeros'),
                Layer(30, 30, 'hidden',
                      activation_func=ActivationFunction.leaky_relu,
                      activation_params={'alpha': 0.01},
                      weight_init='he',
                      bias_init='zeros'),
                Layer(30, 4, 'output',
                      activation_func=ActivationFunction.tanh)
            ],
            'data_file': 'quadrant_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'learning_rate': 0.001,
            'num_epochs': 1500,
            'num_samples': 900,
            'cost_function': 'mae',
            'save_file': 'model_optimized_quadrant.json'
        },

        ############################################################################################################
        # CONFIG 6: HOUSE REGRESSION - REGRESSION OPTIMIZED
        ############################################################################################################
        6: {
            'name': 'House Price Regression (OPTIMIZED)',
            'description': 'Predict house prices (regression with unbounded outputs)',
            'architecture': '3 → 20 → 20 → 20 → 1',
            'details': 'Leaky ReLU throughout + He init + Linear output + MAE',
            'layers': lambda: [
                Layer(3, 3, 'input'),
                Layer(3, 20, 'hidden',
                      activation_func=ActivationFunction.leaky_relu,
                      activation_params={'alpha': 0.01},
                      weight_init='he',
                      bias_init='zeros'),
                Layer(20, 20, 'hidden',
                      activation_func=ActivationFunction.leaky_relu,
                      activation_params={'alpha': 0.01},
                      weight_init='he',
                      bias_init='zeros'),
                Layer(20, 20, 'hidden',
                      activation_func=ActivationFunction.leaky_relu,
                      activation_params={'alpha': 0.01},
                      weight_init='he',
                      bias_init='zeros'),
                Layer(20, 1, 'output',
                      activation_func=ActivationFunction.linear)
            ],
            'data_file': 'linear_regression_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'learning_rate': 0.001,
            'num_epochs': 2000,
            'num_samples': 900,
            'cost_function': 'mae',
            'save_file': 'model_optimized_linear_regression.json'
        },

        ############################################################################################################
        # CONFIG 7: IRIS - MULTI-CLASS SOFTMAX OPTIMIZED (3 classes)
        ############################################################################################################
        7: {
            'name': 'Iris Flower Classification (OPTIMIZED)',
            'description': 'Classify iris flowers into 3 species',
            'architecture': '4 → 30 → 30 → 30 → 3',
            'details': 'Leaky ReLU throughout + He init + Softmax output + Categorical CE',
            'layers': lambda: [
                Layer(4, 4, 'input'),
                Layer(4, 30, 'hidden',
                      activation_func=ActivationFunction.leaky_relu,
                      activation_params={'alpha': 0.01},
                      weight_init='he',
                      bias_init='zeros'),
                Layer(30, 30, 'hidden',
                      activation_func=ActivationFunction.leaky_relu,
                      activation_params={'alpha': 0.01},
                      weight_init='he',
                      bias_init='zeros'),
                Layer(30, 30, 'hidden',
                      activation_func=ActivationFunction.leaky_relu,
                      activation_params={'alpha': 0.01},
                      weight_init='he',
                      bias_init='zeros'),
                Layer(30, 3, 'output',
                      activation_func=ActivationFunction.softmax)
            ],
            'data_file': 'iris_data.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'learning_rate': 0.001,
            'num_epochs': 2000,
            'num_samples': 850,
            'cost_function': 'categorical_crossentropy',
            'save_file': 'model_optimized_iris.json'
        },

        ############################################################################################################
        # MNIST - DIGIT CLASSIFICATION
        ############################################################################################################
        8: {
            'name': 'MNIST Digit Classification',
            'description': 'Classify MNIST digits (0-9) from 28x28 grayscale images',
            'architecture': '784 → 128 → 64 → 10',
            'details': 'Leaky ReLU hidden layers + He init + Softmax output + Categorical CE',
            'layers': lambda: [
                Layer(784, 784, 'input'),
                Layer(784, 128, 'hidden',
                      activation_func=ActivationFunction.leaky_relu,
                      activation_params={'alpha': 0.01},
                      weight_init='he',
                      bias_init='zeros'),
                Layer(128, 64, 'hidden',
                      activation_func=ActivationFunction.leaky_relu,
                      activation_params={'alpha': 0.01},
                      weight_init='he',
                      bias_init='zeros'),
                Layer(64, 10, 'output',
                      activation_func=ActivationFunction.softmax)
            ],
            'data_file': 'mnist_train.json',
            'input_key': 'Input_Values',
            'output_key': 'Output_Values',
            'learning_rate': 0.001,
            'num_epochs': 500,
            'num_samples': 1000,
            'cost_function': 'categorical_crossentropy',
            'save_file': 'model_mnist.json'
        },
    }

    if config_num not in configs:
        raise ValueError(f"Invalid configuration number: {config_num}. Must be 1-8.")

    return configs[config_num]


############################################################################################################
# MAIN EXECUTION
############################################################################################################

# Get the selected configuration
config = get_configuration(CONFIG_TO_RUN)

# Print configuration info
print("=" * 70)
print(f"TRAINING (OPTIMIZED): {config['name']}")
print("=" * 70)
print(f"Description: {config['description']}")
print(f"Architecture: {config['architecture']}")
print(f"Details: {config['details']}")
print("=" * 70)
print()

# Build the neural network
neural_net = NeuralNet()
for layer in config['layers']():
    neural_net.add_layer(layer)

# Load training data
data_file = os.path.join(os.path.dirname(__file__), "data", config['data_file'])
with open(data_file, "r") as file:
    data = json.load(file)

input_data = np.array(data[config['input_key']])
target_data = np.array(data[config['output_key']])

print(f"Loaded {len(input_data)} training samples from {config['data_file']}")
print(f"Learning rate: {config['learning_rate']}")
print(f"Epochs: {config['num_epochs']}")
print(f"Cost function: {config['cost_function']}")
print()

# Create a Training object with checkpointing (saves best model to save_file)
save_file_path = os.path.join(os.path.dirname(__file__), "models", config['save_file'])
training = Training(neural_net, learning_rate=config['learning_rate'], clip_value=5, 
                    cost_function=config['cost_function'], checkpoint_path=save_file_path, batch_size=config.get('batch_size', 1))

# Train the neural network
training.train(input_data, target_data, epochs=config['num_epochs'], samples_per_epoch=config['num_samples'])

# Best model already saved during training via checkpoint
print()
print("=" * 70)
print(f"✅ Training complete! Best model saved to {config['save_file']}")
print(f"📊 Best cost achieved: {training.best_cost:.6f}")
print("=" * 70)