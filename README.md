# Shitty-Python-Neural-Net
read the title lmao

more concretely, this is a framework for building multilayer perceptrons

I don't plan to update this in the forseeable future. Pull requests/issues welcome (assuming anyone even comes across this repo lmao).

# Todo
 - [x] Make the someActivationFunction.derivative thing work instead of manually setting it in layers
    - now automatically detects and sets derivatives
 - [x] Instead of MSE cost, use cross entropy
    - added `cost_function` parameter with MSE, MAE, and bin/cat CE options
 - [x] Perhaps instead of tanh as the output layer activation function, use softmax, or maybe even sigmoid
    - output activation is no longer hardcoded, defaults to tanh but can be overridden with softmax, sigmoid, or any activation
 - [x] Allow the training to choose a certain subset of the total data to train with for a single epoch
    - added `samples_per_epoch` parameter to randomly sample a subset each epoch
 - [x] Optimize double forward propagation in training
    - backprop now returns both gradients and predictions so during training, backprop doesn't have to run twice
 - [x] Allow custom alpha values for leaky relu (currently hardcoded to 0.01)
    - now supports activation_params dict for all parametric activations, and weight_init_params/bias_init_params for initializers

# Known Issues
If you set...
 - the learning rate too high (>0.001)
 - too high of a clipping barrier (haven't tested but its a given because when I didn't clip, it just killed itself)
 - or initialize layer weights to be too large (I initially did it with a normal distr mean0 and std1, but I had to lower the std)

the network will diverge due to vanishing or exploding gradients (see `/docs/GRADIENT_ANALYSIS.md` for additional details). This apparently is a common issue with neural networks, and is usually solved by clipping the network or by using a lower learning rate.

**He/Xavier initialization requires normalized data!** If you use `weight_init='he'` or `weight_init='xavier'` with unnormalized data (e.g., raw RGB 0-255), the large inputs × large weights = exploding activations and training fails. Either normalize your data first (divide RGB by 255.0 to get 0-1 range), or use a weight init like `weight_init='normal', weight_init_params={'std': 0.01}`.

If the overall dataset size is small, the network may overfit. For example, the RGB classification problem has only 256<sup>3</sup> possible inputs. Use `samples_per_epoch` to train on a random subset each epoch for regularization. I don't know if a supposed "grokking" phenomenon would occur, so further testing is needed.

The network initializes weights and biases randomly, so training results may vary between runs. So I guess you can just initialize multiple times and pick the best one, then continue training from there. You could also set a random seed for reproducibility if desired (I haven't implemented, but you can set `np.random.seed(your_seed)` at the start of your script). 

# Usage
Run the main scripts lmao and change them as you'd like to make your own MLP neural net.
<img>https://media.tenor.com/E_41h0i3pdAAAAAe/fluttershy-mlp.png</img>

## Basic Training
```python
# Train on all data each epoch (default behavior)
training.train(input_data, target_data, epochs=500)
```

## Subset Training (Recommended for Larger Datasets)
```python
# Train on 400 randomly selected samples per epoch (out of 1000 total)
# This helps prevent overfitting and adds regularization
training.train(input_data, target_data, epochs=500, samples_per_epoch=400)
```

**Benefits of subset training:**
- Prevents overfitting on small/limited datasets
- Adds regularization through data sampling
- Each epoch sees different random samples
- Faster training per epoch

## Automatic Best Model Checkpointing (NEW!)

**All training scripts now automatically save the best model during training!**

```python
# Checkpointing is enabled by default in main_create_and_train.py and main_continue_training.py
# The best model (lowest cost) is automatically saved during training

training = Training(neural_net,
                   learning_rate=0.001,
                   clip_value=5,
                   cost_function='mse',
                   checkpoint_path='models/model_best.json')  # Auto-saves best here

training.train(input_data, target_data, epochs=5000)
# Models saved automatically whenever cost improves!
```

**What you'll see:**
```
Epoch 1/5000, Average Cost: 0.543
Epoch 2/5000, Average Cost: 0.421 - NEW BEST! Saved to models/model_sine.json
Epoch 3/5000, Average Cost: 0.398 - NEW BEST! Saved to models/model_sine.json
...
Epoch 150/5000, Average Cost: 0.082 - NEW BEST! Saved to models/model_sine.json
Epoch 151/5000, Average Cost: 0.510  ← Cost exploded, but 0.082 is safe!
...
Training complete! Best model saved to model_sine.json
Best cost achieved: 0.082
```

**Why this is critical:**
- **Prevents loss from overfitting** - Cost may rise after epoch 500 but you keep the best from epoch 150
- **Prevents loss from learning rate spikes** - If cost explodes (0.08 → 0.5), you don't lose the good model
- **No manual intervention needed** - Automatic safety net for long training runs

**To disable checkpointing** (not recommended):
```python
training = Training(neural_net, learning_rate=0.001, clip_value=5, cost_function='mse')
# Omit checkpoint_path parameter - no automatic saving
```

# Notes
Numpy is the only dependency.

An epoch is one iteration through the entire training set. You may generate new training sets but I haven't tested the net much with this thing because it seems to get stuck at some fixed cost and refuse to make any further progress lmao.

Currently, the network is being trained to recongnize if a given RGB color triple is "red" or not, as defined in my definition of red python script which also generates the training data. The code is written noobishly so optimization would be nice.

# Repository Structure

```
/
├── main_create_and_train.py    # Train new models
├── main_load.py                # Evaluate trained models
├── main_continue_training.py   # Fine-tune existing models
├── README.md
│
├── src/                        # Core neural network code
│   ├── activation_functions.py # Activation functions and derivatives
│   ├── layer.py                # Layer implementation
│   ├── neural_network.py       # Neural network class
│   └── training.py             # Training and backpropagation
│
├── data_generators/            # Dataset generation scripts
│   ├── definition_of_red.py
│   ├── definition_of_xor.py
│   ├── definition_of_sine.py
│   ├── definition_of_checkerboard.py
│   └── definition_of_quadrant.py
│
├── data/                       # Generated datasets (JSON)
│   ├── color_data.json
│   ├── xor_data.json
│   ├── sine_data.json
│   ├── checkerboard_data.json
│   └── quadrant_data.json
│
├── models/                     # Trained models (JSON)
│   ├── model_red.json       # RGB classifier
│   ├── model_xor.json
│   ├── model_sine.json
│   ├── model_checkerboard.json
│   └── model_quadrant.json
│
└── docs/                       # Documentation
    ├── GRADIENT_ANALYSIS.md
    └── TRAINING_GUIDE.md
```

---
# Activation Functions Guide
 
## Quick Summary

- **Input layers:** No activation (just passes data through)
- **Hidden layers:** Leaky ReLU (default, alpha=0.01) - alternatives: ReLU, ELU
  - Note: Sigmoid/Tanh can be used but suffer from vanishing gradients in deep networks
- **Output layers:** Tanh (default) - alternatives: Sigmoid (binary classification), Softmax (multi-class classification), Linear (regression)
  - **Linear (identity)** is now available for regression with unbounded outputs
 
## Available Activation Functions
 
### Sigmoid
- **Formula:** `f(x) = 1 / (1 + e^(-x))`
- **Range:** (0, 1)
- **When to use:** Output layer for binary classification (0 or 1 probabilities)
- **Issues:** Suffers from vanishing gradients, saturates easily
- **Usage:**
  ```python
  layer = Layer(10, 5, 'hidden', activation_func=ActivationFunction.sigmoid)
  ```
 
### Tanh (Hyperbolic Tangent)
- **Formula:** `f(x) = tanh(x)`
- **Range:** (-1, 1)
- **When to use:** Output layer (currently hardcoded for all output layers), or hidden layers if you want zero-centered outputs
- **Issues:** Still has vanishing gradient issues but better than sigmoid
- **Why we use it:** Normalizes outputs to [-1, 1] which is what our training data uses
- **Usage:** Automatically used for output layers, or:
  ```python
  layer = Layer(10, 5, 'hidden', activation_func=ActivationFunction.tanh)
  ```
 
### ReLU (Rectified Linear Unit)
- **Formula:** `f(x) = max(0, x)`
- **Range:** [0, ∞)
- **When to use:** Fast training, simple problems, when you don't care about "dying neurons"
- **Issues:** Dying ReLU problem - neurons can get stuck at 0 and stop learning
- **Usage:**
  ```python
  layer = Layer(10, 5, 'hidden', activation_func=ActivationFunction.relu)
  ```
 
### Leaky ReLU (Default for Hidden Layers)
- **Formula:** `f(x) = max(alpha * x, x)` where alpha=0.01 by default
- **Range:** (-∞, ∞)
- **When to use:** Almost always for hidden layers (it's the default!)
- **Why it's good:** Fixes dying ReLU problem - neurons never completely die because negative values get small gradient (alpha)
- **Customization:**
  ```python
  # Default alpha=0.01
  layer = Layer(10, 5, 'hidden')
 
  # Custom alpha
  layer = Layer(10, 5, 'hidden',
                activation_func=ActivationFunction.leaky_relu,
                activation_params={'alpha': 0.05})
  ```
 
### ELU (Exponential Linear Unit)
- **Formula:**
  - `f(x) = x` when x > 0
  - `f(x) = alpha * (e^x - 1)` when x ≤ 0
- **Range:** (-alpha, ∞)
- **When to use:** When you want smoother gradients than leaky ReLU, can help training converge faster
- **Why it's good:** Smoother than leaky ReLU, pushes mean activations closer to zero which can speed up learning
- **Customization:**
  ```python
  # Default alpha=1.0
  layer = Layer(10, 5, 'hidden', activation_func=ActivationFunction.elu)
 
  # Custom alpha
  layer = Layer(10, 5, 'hidden',
                activation_func=ActivationFunction.elu,
                activation_params={'alpha': 0.5})
  ```
 
### Linear (Identity Function)
- **Formula:** `f(x) = x`
- **Range:** (-∞, ∞) - unbounded
- **When to use:** Output layer for regression with unbounded targets (house prices, temperatures, stock prices, etc.)
- **Why it's good:** Allows network to output any value, not constrained to a specific range
- **Derivative:** `f'(x) = 1` - gradient passes through unchanged, very stable
- **Usage:**
  ```python
  # For regression predicting continuous unbounded values
  output_layer = Layer(5, 1, 'output', activation_func=ActivationFunction.linear)

  # Use with MSE or MAE cost functions
  training = Training(neural_net, learning_rate=0.001, clip_value=1, cost_function='mse')
  ```

### Sign and Step
- **Sign:** Returns -1, 0, or 1 based on sign of input
- **Step:** Returns 0 or 1 based on threshold
- **When to use:** Almost never lmao, gradients are zero everywhere so backprop doesn't work
- **Why they exist:** Historical/theoretical purposes
- **Usage:** Don't use these unless you know what you're doing
 
## Layer-Specific Recommendations
 
### Input Layers
```python
input_layer = Layer(3, 3, 'input')
```
- **Activation:** None (just passes data through)
- **Why:** Input layer just represents the raw input data, no transformation needed
- **Weights/biases:** Initialized to zero and ignored
 
### Hidden Layers
```python
# Default (recommended for most cases)
hidden_layer = Layer(10, 5, 'hidden')  # Uses Leaky ReLU alpha=0.01
 
# Custom activation
hidden_layer = Layer(10, 5, 'hidden', activation_func=ActivationFunction.elu)
 
# Custom parameters
hidden_layer = Layer(10, 5, 'hidden',
                     activation_func=ActivationFunction.leaky_relu,
                     activation_params={'alpha': 0.02})
```
 
**Recommendations:**
- **Default choice:** Leaky ReLU (it's what we use everywhere and it works well)
- **For faster convergence:** Try ELU
- **For simple problems:** ReLU is fine
- **Avoid:** Sigmoid and Tanh in deep networks (vanishing gradients)
 
### Output Layers
```python
# Default: uses Tanh
output_layer = Layer(5, 2, 'output')

# Override with Sigmoid for binary classification
output_layer = Layer(5, 2, 'output', activation_func=ActivationFunction.sigmoid)

# Override with Softmax for multi-class classification
output_layer = Layer(5, 4, 'output', activation_func=ActivationFunction.softmax)
```

**Default behavior:**
- **Defaults to Tanh** but can be overridden
- **Why Tanh default:** Our training data uses outputs in [-1, 1] range (e.g., `[1, -1]` for red, `[-1, 1]` for not-red)
- **Full flexibility:** You can now use any activation function for output layers!

**Available alternatives:**
- **Sigmoid:** For binary classification with 0/1 outputs
  - Range: [0, 1]
  - Use with: Binary Cross-Entropy loss
  - Data format: `[1, 0]` for class A, `[0, 1]` for class B
- **Softmax:** For multi-class classification with probability distributions
  - Range: [0, 1] with outputs summing to 1
  - Use with: Categorical Cross-Entropy loss (REQUIRED - see Cost Functions Guide)
  - Data format: One-hot encoded like `[1, 0, 0, 0]` for class 1
  - Example: Instead of `[1, -1, -1, -1]` for quadrant 1, softmax would output `[0.97, 0.01, 0.01, 0.01]` (probabilities)
- **Tanh:** For regression or classification with [-1, 1] data
  - Range: [-1, 1]
  - Use with: MSE or MAE loss
  - Data format: `[1, -1]` for class A, `[-1, 1]` for class B
- **Linear (identity function):** For regression with unbounded outputs
  - Range: (-∞, ∞) - unbounded
  - Formula: `f(x) = x` (no transformation applied)
  - Derivative: `f'(x) = 1` (gradient passes through unchanged)
  - Use with: MSE or MAE loss
  - Example: Predicting house prices ($100k-$1M), temperatures (-20°C to 40°C), stock prices, etc.
  - Usage:
    ```python
    output_layer = Layer(5, 1, 'output', activation_func=ActivationFunction.linear)
    training = Training(neural_net, learning_rate=0.001, clip_value=1, cost_function='mse')
    ```
 
## Mix and Match Example
 
You can use different activation functions for different layers:
 
```python
neural_net = NeuralNet()
 
# Input layer (no activation)
neural_net.add_layer(Layer(3, 3, 'input'))
 
# First hidden layer - ELU with custom alpha
neural_net.add_layer(Layer(3, 10, 'hidden',
                           activation_func=ActivationFunction.elu,
                           activation_params={'alpha': 0.8}))
 
# Second hidden layer - Leaky ReLU with custom alpha
neural_net.add_layer(Layer(10, 5, 'hidden',
                           activation_func=ActivationFunction.leaky_relu,
                           activation_params={'alpha': 0.02}))
 
# Third hidden layer - Standard ReLU
neural_net.add_layer(Layer(5, 4, 'hidden',
                           activation_func=ActivationFunction.relu))
 
# Output layer (automatically uses Tanh)
neural_net.add_layer(Layer(4, 2, 'output'))
```
 
This flexibility lets you experiment to see what works best for your problem!

-----
# Weight/Bias Initialization Guide

## Quick Summary

Weight initialization determines the starting values of your neural network's weights and biases. Poor initialization can cause vanishing/exploding gradients or slow training.

**Default settings (recommended):**
```python
layer = Layer(10, 5, 'hidden')  # Uses weight_init='he', bias_init='zeros'
```

## Available Weight Initializers

### Normal Distribution
```python
layer = Layer(10, 5, 'hidden',
              weight_init='normal',
              weight_init_params={'std': 0.01})
```
- **Formula:** Sample from N(0, std²)
- **Default std:** 0.01
- **When to use:** With unnormalized data (e.g., RGB 0-255), or when you need very small weights
- **Pros:** Simple, works with raw data
- **Cons:** Not optimal - fixed std doesn't scale with layer size

### Xavier/Glorot Initialization
```python
layer = Layer(10, 5, 'hidden',
              weight_init='xavier',
              weight_init_params={})  # No params needed
```
- **Formula:** std = sqrt(2 / (fan_in + fan_out))
- **When to use:** Sigmoid or Tanh activations in hidden layers
- **Pros:** Keeps variance stable across layers
- **Cons:** Not ideal for ReLU-like activations
- **⚠️ REQUIRES NORMALIZED DATA** (see Known Issues)

### He Initialization (Default)
```python
layer = Layer(10, 5, 'hidden',
              weight_init='he',
              weight_init_params={})  # No params needed
```
- **Formula:** std = sqrt(2 / fan_in)
- **When to use:** ReLU, Leaky ReLU, ELU activations (default is Leaky ReLU!)
- **Pros:** Best for ReLU-like activations, accounts for dead neurons
- **Cons:** Can cause issues with unnormalized data
- **⚠️ REQUIRES NORMALIZED DATA** (see Known Issues)
- **Note:** This is the industry standard for modern networks with ReLU/Leaky ReLU

### Uniform Distribution
```python
layer = Layer(10, 5, 'hidden',
              weight_init='uniform',
              weight_init_params={'limit': 0.1})
```
- **Formula:** Sample uniformly from [-limit, limit]
- **When to use:** Rarely - prefer normal distributions in practice
- **Requires:** You must specify `limit` parameter

### Uniform Xavier
```python
layer = Layer(10, 5, 'hidden',
              weight_init='uniform_xavier',
              weight_init_params={})
```
- **Formula:** limit = sqrt(6 / (fan_in + fan_out))
- **When to use:** Uniform variant of Xavier initialization
- **⚠️ REQUIRES NORMALIZED DATA**

## Available Bias Initializers

### Zeros (Default)
```python
layer = Layer(10, 5, 'hidden',
              bias_init='zeros',
              bias_init_params={})
```
- **Most common approach** - biases start at 0
- **Why it works:** Weights break symmetry, biases can start at 0

### Ones
```python
layer = Layer(10, 5, 'hidden',
              bias_init='ones',
              bias_init_params={})
```
- **Rarely used** - all biases set to 1
- **When to use:** Almost never in practice

### Constant
```python
layer = Layer(10, 5, 'hidden',
              bias_init='constant',
              bias_init_params={'value': 0.5})
```
- **Use case:** LSTM forget gates (often set to 1.0)
- **Requires:** You must specify `value` parameter

### Normal Distribution
```python
layer = Layer(10, 5, 'hidden',
              bias_init='normal',
              bias_init_params={'std': 0.01})
```
- **Rarely used** - zeros is almost always better
- **Default std:** 0.01 if used

## Recommendations by Activation Function

| Activation Function | Best Weight Init | Data Requirement |
|---------------------|------------------|------------------|
| Leaky ReLU (default) | `he` (default) | Normalized (0-1 or mean=0, std=1) |
| ReLU | `he` | Normalized |
| ELU | `he` | Normalized |
| Sigmoid | `xavier` | Normalized |
| Tanh | `xavier` | Normalized |
| Any (unnormalized data) | `normal` with std=0.01 | No normalization needed |

## Data Normalization Requirements

**CRITICAL:** He and Xavier initialization assume **normalized input data** (mean≈0, std≈1).

### For RGB Data (0-255):
```python
# In your data generator:
data_entry_1 = [[r/255.0, g/255.0, b/255.0] for r, g, b in all_samples]
```

### For General Data:
```python
import numpy as np
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
normalized_data = (data - mean) / std
```

### What Happens Without Normalization:
- Raw RGB (0-255) × He weights (std≈0.8) = activations in range [-200, 200]
- Tanh/Sigmoid saturate completely (outputs stuck at ±1)
- Gradients vanish (derivative ≈ 0)
- Training fails (cost stuck at 1.0 or explodes to NaN)

### Workaround (Not Recommended):
```python
# If you can't normalize data, use small fixed weights:
layer = Layer(10, 5, 'hidden',
              weight_init='normal',
              weight_init_params={'std': 0.01})
```
This works but isn't optimal - doesn't scale properly with layer size.

## Full Example

```python
neural_net = NeuralNet()

# Input layer (weights unused)
neural_net.add_layer(Layer(3, 3, 'input'))

# Hidden layer with He init (default - best for Leaky ReLU with normalized data)
neural_net.add_layer(Layer(3, 10, 'hidden'))

# Hidden layer with custom normal init (for unnormalized data)
neural_net.add_layer(Layer(10, 5, 'hidden',
                           weight_init='normal',
                           weight_init_params={'std': 0.02},
                           bias_init='constant',
                           bias_init_params={'value': 0.1}))

# Output layer (defaults to Tanh activation, but can be overridden)
neural_net.add_layer(Layer(5, 2, 'output'))
```

-----
# Cost Functions Guide

## Quick Summary

**Default:** MSE (Mean Squared Error) - works with any output activation

**Available cost functions:**
- `'mse'` - Mean Squared Error (default)
- `'mae'` - Mean Absolute Error
- `'binary_crossentropy'` - Binary Cross-Entropy (for binary classification)
- `'categorical_crossentropy'` - Categorical Cross-Entropy (for multi-class classification)

## How to Use

```python
# Create training object with chosen cost function
training = Training(neural_net,
                   learning_rate=0.001,
                   clip_value=1,
                   cost_function='mse')  # Change this parameter

# Train as normal
training.train(input_data, target_data, epochs=500)
```

## Available Cost Functions

### Mean Squared Error (MSE) - Default
```python
training = Training(neural_net, learning_rate=0.001, clip_value=1, cost_function='mse')
```
- **Formula:** `0.5 * (predicted - target)²`
- **When to use:** General-purpose, works with any output activation and data range
- **Works with:** Tanh [-1, 1], Sigmoid [0, 1], Softmax, or any activation
- **Data format:** Any range (e.g., [-1, 1] or [0, 1])
- **Pros:** Smooth gradients, penalizes large errors more heavily
- **Cons:** Sensitive to outliers

### Mean Absolute Error (MAE)
```python
training = Training(neural_net, learning_rate=0.001, clip_value=1, cost_function='mae')
```
- **Formula:** `|predicted - target|`
- **When to use:** When you want to be less sensitive to outliers than MSE
- **Works with:** Tanh [-1, 1], Sigmoid [0, 1], Softmax, or any activation
- **Data format:** Any range (e.g., [-1, 1] or [0, 1])
- **Pros:** Less sensitive to outliers, more robust
- **Cons:** Gradient is not smooth at predicted = target

### Binary Cross-Entropy
```python
training = Training(neural_net, learning_rate=0.001, clip_value=1, cost_function='binary_crossentropy')
```
- **Formula:** `-[target * log(predicted) + (1-target) * log(1-predicted)]`
- **When to use:** Binary classification (two classes only)
- **REQUIRES:** Sigmoid output activation [0, 1]
- **Data format:** Targets **MUST** be 0 or 1 (e.g., `[1, 0]` for class A, `[0, 1]` for class B)
- **Pros:** Standard for binary classification, works well with sigmoid
- **Cons:** Requires specific data format

**⚠️ CRITICAL: Binary Cross-Entropy Requires [0, 1] Target Data!**

If you use Binary CE with `[-1, 1]` target data (common in this repo), you will get **negative costs** and completely wrong gradients!

```python
# WRONG - Will produce negative costs and broken training:
target_data = [[1, -1], [-1, 1], ...]  # [-1, 1] format
training = Training(net, ..., cost_function='binary_crossentropy')  # ❌ BAD!

# CORRECT - Use [0, 1] format:
target_data = [[1, 0], [0, 1], ...]  # [0, 1] format
training = Training(net, ..., cost_function='binary_crossentropy')  # ✅ GOOD!

# OR - Use MSE/MAE with [-1, 1] data instead:
target_data = [[1, -1], [-1, 1], ...]  # [-1, 1] format is fine
training = Training(net, ..., cost_function='mse')  # ✅ GOOD!
```

**Why this matters:**
- Binary CE formula assumes probabilities (0 to 1 range)
- Negative targets (-1) cause negative log values → negative cost
- Gradients point in wrong direction → network learns nothing or diverges
- **Always check your data format matches your cost function!**

**Example setup:**
```python
neural_net = NeuralNet()
neural_net.add_layer(Layer(3, 3, 'input'))
neural_net.add_layer(Layer(3, 10, 'hidden'))
# Override output to use sigmoid instead of default tanh
neural_net.add_layer(Layer(10, 2, 'output', activation_func=ActivationFunction.sigmoid))

# Make sure your target data is [0, 1] format!
training = Training(neural_net, learning_rate=0.001, clip_value=1, cost_function='binary_crossentropy')
```

### Categorical Cross-Entropy
```python
training = Training(neural_net, learning_rate=0.001, clip_value=1, cost_function='categorical_crossentropy')
```
- **Formula:** `-sum(target * log(predicted))`
- **When to use:** Multi-class classification (3+ classes)
- **Best with:** Softmax output activation (probability distribution)
- **Data format:** Targets should be one-hot encoded (e.g., `[1, 0, 0, 0]` for class 1)
- **Pros:** Standard for multi-class classification, works perfectly with softmax
- **Cons:** Requires one-hot encoded targets and softmax output

**Example setup:**
```python
neural_net = NeuralNet()
neural_net.add_layer(Layer(3, 3, 'input'))
neural_net.add_layer(Layer(3, 10, 'hidden'))
# Override output to use softmax instead of default tanh
neural_net.add_layer(Layer(10, 4, 'output', activation_func=ActivationFunction.softmax))

training = Training(neural_net, learning_rate=0.001, clip_value=1, cost_function='categorical_crossentropy')
```

## Cost Function & Output Activation Compatibility

**IMPORTANT:** This compatibility matrix is for **OUTPUT LAYER** activations only!

### Activation Functions - Where They Can Be Used

**Softmax - OUTPUT LAYERS ONLY (hardcoded restriction):**
- **Must be used on output layer only** due to hardcoded special case in backpropagation
- For multi-class classification (probability distributions)
- MUST use with Categorical Cross-Entropy

**All Other Activations - Can Be Used Anywhere (Hidden OR Output):**
- **Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Linear** - No hardcoded restrictions, can be used on any layer
- **Typical usage patterns:**
  - **Sigmoid/Tanh**: Commonly used on output for classification, rarely on hidden (vanishing gradients)
  - **ReLU/Leaky ReLU/ELU**: Commonly used on hidden layers, can be used on output for regression
  - **Linear**: Primarily for regression output (unbounded), rarely on hidden (no non-linearity)
- But technically all work on both hidden and output layers

### Recommended Combinations (What Works Correctly)

| Cost Function | Compatible **Output** Activations | Data Format | Use Case |
|---------------|----------------------|-------------|----------|
| **MSE** | Tanh, Sigmoid, ReLU, Leaky ReLU, ELU, Linear | Any range | General-purpose, regression, classification with [-1, 1] or [0, 1] data |
| **MAE** | Tanh, Sigmoid, ReLU, Leaky ReLU, ELU, Linear | Any range | General-purpose, robust to outliers |
| **Binary Cross-Entropy** | **Sigmoid ONLY** | Targets: 0 or 1 | Binary classification (2 classes) |
| **Categorical Cross-Entropy** | **Softmax ONLY** | Targets: one-hot encoded | Multi-class classification (3+ classes) |

### Detailed Compatibility Matrix (Output Layer Only)

| **Output** Activation | MSE | MAE | Binary CE | Categorical CE | Typical Use |
|------------|-----|-----|-----------|----------------|-------------|
| **Tanh** | ✅ Works | ✅ Works | ❌ **WRONG** (negative outputs clipped) | ❌ **WRONG** (negative outputs clipped) | Classification with [-1, 1] data |
| **Sigmoid** | ✅ Works | ✅ Works | ✅ **RECOMMENDED** | ⚠️ **WRONG** (sum ≠ 1, not a distribution) | Binary classification |
| **Softmax** | ❌ **WRONG GRADIENTS** | ❌ **WRONG GRADIENTS** | ❌ **WRONG GRADIENTS** | ✅ **REQUIRED** | Multi-class classification |
| **Linear** | ✅ **RECOMMENDED** | ✅ **RECOMMENDED** | ❌ **WRONG** (unbounded, not [0, 1]) | ❌ **WRONG** (unbounded, not [0, 1]) | **Regression** (unbounded outputs) |
| **ReLU/Leaky ReLU/ELU** | ✅ Works | ✅ Works | ❌ **WRONG** (unbounded, not [0, 1]) | ❌ **WRONG** (unbounded, not [0, 1]) | Regression (but Linear is better) |

**Notes:**
- **Softmax**: Only activation with hardcoded restrictions - MUST be on output layer with Categorical CE
- **All others**: Can technically be used anywhere, but typical usage patterns vary (see above)

**Legend:**
- ✅ Works correctly
- ⚠️ Runs but mathematically incorrect
- ❌ Produces wrong results (may or may not crash)

## Important Notes

### Output Layer Activation Defaults
- **Output layers now default to Tanh** (outputs in [-1, 1] range)
- **You can override** by passing `activation_func` parameter
- **No longer hardcoded** - full flexibility!

```python
# Default: uses Tanh
output_layer = Layer(5, 2, 'output')

# Override with Sigmoid for Binary Cross-Entropy
output_layer = Layer(5, 2, 'output', activation_func=ActivationFunction.sigmoid)

# Override with Softmax for Categorical Cross-Entropy
output_layer = Layer(5, 4, 'output', activation_func=ActivationFunction.softmax)
```

### Data Format Requirements
- **MSE/MAE with Tanh:** Use `[-1, 1]` range (e.g., `[1, -1]` for red, `[-1, 1]` for not-red)
- **Binary Cross-Entropy with Sigmoid:** Use `[0, 1]` range (e.g., `[1, 0]` for class A, `[0, 1]` for class B)
- **Categorical Cross-Entropy with Softmax:** Use one-hot encoding (e.g., `[1, 0, 0, 0]` for class 1)

### Testing with Current Data
The current data generators use `[-1, 1]` format, which works perfectly with:
- MSE + Tanh (default setup)
- MAE + Tanh

To test Binary/Categorical Cross-Entropy, you would need to:
1. Regenerate data in `[0, 1]` or one-hot format
2. Change output activation to Sigmoid or Softmax
3. Use the corresponding cost function

### Numerical Stability
Both cross-entropy cost functions include automatic clipping (epsilon = 1e-15) to prevent `log(0)` errors. This ensures stable training even with extreme predictions.

## Optimized Backpropagation (Special Cases)

### Softmax + Categorical Cross-Entropy (Hardcoded Optimization)

When you use **softmax output activation** with **categorical cross-entropy loss**, the backpropagation automatically detects this combination and uses a **hardcoded optimization**:

```
gradient = predicted_values - target_values
```

**Why this is REQUIRED (not just an optimization):**
- Softmax has a Jacobian matrix derivative (not element-wise like other activations)
- Cannot use standard element-wise multiplication: `(dCost/dPredicted) * (dPredicted/dZ)`
- The full Jacobian-vector product simplifies to: `predicted - target`
- This is a well-known result in deep learning and is numerically stable

**Detection:** Automatic - if `output_layer.activation_func.title == 'softmax'` and `cost_function == 'categorical_crossentropy'`

### All Other Combinations (Element-wise)

For **all other activation + cost function combinations**, backpropagation uses the standard element-wise chain rule:

```
gradient = (dCost/dPredicted) * (dPredicted/dZ)
```

This works correctly for element-wise activations like:
- **Sigmoid** (including sigmoid + binary cross-entropy, which naturally simplifies to `predicted - target` through the multiplication)
- **Tanh** (with MSE, MAE, or any cost function)
- **ReLU, Leaky ReLU, ELU** (with any cost function)

**Sigmoid + Binary Cross-Entropy note:**
- Uses the general element-wise case (no special hardcoded optimization needed)
- BCE derivative: `(predicted - target) / [predicted × (1 - predicted)]`
- Sigmoid derivative: `predicted × (1 - predicted)`
- When multiplied: terms naturally cancel to give `predicted - target`

### Implementation Note
The softmax + CE special case is implemented in `src/training.py` in the `firstTwoDerivativesOfOutputLayer()` method. The detection is automatic - you don't need to do anything special. Just choose the right cost function and output activation.

### Switching Cost Functions During Training

**You CAN switch cost functions when continuing training from a saved model:**

```python
# Initial training with MAE
trainer = Training(net, learning_rate=0.01, clip_value=5, cost_function='mae')
trainer.train(input_data, target_data, epochs=500)
net.save('model.json')

# Continue training with MSE
loaded_net = NeuralNet()
loaded_net.load('model.json')
trainer2 = Training(loaded_net, learning_rate=0.01, clip_value=5, cost_function='mse')  # Different!
trainer2.train(input_data, target_data, epochs=500)
```

**Why this works:**
- Cost functions are properties of the `Training` object, not the `NeuralNet`
- Saved models only contain weights, biases, and architecture - no cost function
- When you create a new `Training` object, you can choose any compatible cost function

**Restrictions:**
- The same activation/cost compatibility rules still apply (see Compatibility Matrix)
- ❌ Can't switch TO binary_crossentropy unless output is sigmoid
- ❌ Can't switch TO categorical_crossentropy unless output is softmax
- ❌ Can't switch FROM softmax to anything except categorical_crossentropy
- ✅ Can freely switch between MSE ⟷ MAE for any activation

**Practical recommendations:**
- ✅ **MSE ⟷ MAE are switchable** - Both are regression losses with different error metrics, switching makes sense
- ❌ **Binary CE and Categorical CE - pick one and stick with it** - These are classification losses with different problem framings, switching mid-training doesn't make practical sense

**Practical use case for MSE ⟷ MAE:**
1. **Start with MAE** (robust to outliers, gets "close enough")
2. **Switch to MSE** (fine-tunes, penalizes large errors more heavily)

This is a legitimate training strategy - MAE is less sensitive to outliers early on, then MSE polishes the fit.

### Important Limitations

**CRITICAL: Refer to the Compatibility Matrix above before choosing combinations!**

**Softmax activation is for OUTPUT LAYERS ONLY and MUST use Categorical Cross-Entropy:**
- **Output layer only:** Softmax is designed for output layers in multi-class classification (technically can be used on hidden layers but serves no purpose)
- **Must use with Categorical CE:** Due to how softmax derivatives work (Jacobian matrix, not element-wise), the current implementation only supports softmax when paired with categorical cross-entropy
- Using softmax with MSE, MAE, or Binary CE will **not crash** but produces **incorrect gradients** silently
- The softmax derivative is a placeholder that returns softmax output, which only works with the hardcoded CE special case
- Training will appear to work but the network won't learn correctly
- This isn't a practical limitation - softmax is designed for probability distributions, and CE is the natural loss for that use case

**Binary Cross-Entropy MUST ONLY be used with Sigmoid:**
- Binary CE is designed for sigmoid activations (outputs in [0, 1] range)
- **Using with Tanh:** Negative outputs get clipped to near-zero positive values, producing incorrect costs and gradients (won't crash due to epsilon clipping)
- **Using with ReLU/Leaky ReLU/ELU:** Unbounded outputs (can be > 1 or < 0) break the probability interpretation
- **Using with Softmax:** Will produce incorrect gradients (use Categorical CE instead)

**Categorical Cross-Entropy MUST ONLY be used with Softmax:**
- Categorical CE is designed for softmax (probability distributions that sum to 1)
- **Using with Sigmoid:** Runs but mathematically wrong - sigmoid outputs are independent probabilities, not a distribution (sum ≠ 1)
- **Using with Tanh:** Negative outputs get clipped, producing incorrect results
- **Using with ReLU/Leaky ReLU/ELU:** Unbounded outputs break the probability interpretation

**Safe output layer combinations (see Compatibility Matrix for full details):**
- **Sigmoid (output):** MSE, MAE, or Binary Cross-Entropy
- **Tanh (output):** MSE or MAE only
- **Softmax (output only):** Categorical Cross-Entropy only
- **Linear (output):** MSE or MAE (for unbounded regression)
- **ReLU, Leaky ReLU, ELU:** Hidden layers only (not for classification output)

---

# Troubleshooting Guide

## Cost Increases During Training

**"Why does my cost go UP when gradient descent is supposed to minimize it?"**

Cost can increase for several reasons - this is normal and expected in many cases!

### 1. Learning Rate Too High (Most Common)

Gradient descent takes steps that are too large and overshoots the minimum:

```
Epoch 100: Cost = 0.08  ← At minimum
Epoch 101: Cost = 0.50  ← Jumped over it!
```

**Symptoms:**
- Sharp spike in cost (0.08 → 0.5 in one epoch)
- Cost oscillates wildly instead of decreasing smoothly
- Network diverges to NaN or infinity

**Solutions:**
- Lower learning rate by 10x: `learning_rate=0.0001` instead of `0.001`
- If continuing training, start with even lower LR: `learning_rate=0.00005`
- Use checkpointing (enabled by default) to recover the best model

### 2. Overfitting

Network starts memorizing training noise instead of learning real patterns:

```
Epoch 1-500: Cost decreases to 0.05  ← Learning
Epoch 501-1000: Cost rises to 0.12   ← Overfitting
```

**Symptoms:**
- Gradual rise in cost after reaching a minimum
- Training cost low but validation cost high
- Network is too large for the dataset

**Solutions:**
- Stop training when cost starts rising (checkpointing saves the best model automatically!)
- Use `samples_per_epoch` for regularization: `train(..., samples_per_epoch=800)`
- Reduce network size (fewer neurons/layers)
- The best model is already saved via checkpoint - just use that!

### 3. Stochastic Noise (NORMAL!)

When using `samples_per_epoch`, each epoch trains on different random samples:

```
Epoch 100: Cost = 0.08  (samples [1, 5, 8, ...])
Epoch 101: Cost = 0.12  (samples [3, 7, 9, ...]) ← Different samples!
Epoch 102: Cost = 0.07  (samples [2, 4, 6, ...])
```

**Symptoms:**
- Cost bounces around instead of smooth decrease
- Fluctuations are relatively small (±0.02 to ±0.05)
- Trend is still generally downward

**This is normal and actually good** - it prevents overfitting! The checkpoint system saves the best model despite the noise.

### 4. Gradient Explosion

Even with clipping, gradients can cause bad updates:

```
Epoch 200: Cost = 0.08
Epoch 201: Cost = 5.23  ← Exploded!
Epoch 202: Cost = NaN   ← Dead network
```

**Symptoms:**
- Cost jumps to very large values (> 1.0)
- Cost becomes NaN or infinity
- Happens suddenly after stable training

**Solutions:**
- Tighter gradient clipping: `clip_value=1` instead of `5`
- Lower learning rate
- Check that data is normalized (divide by 255 for RGB)
- Restart training with better hyperparameters

### 5. Escaping Local Minimum (Good!)

Sometimes cost needs to increase temporarily to find a better minimum:

```
Epoch 100: Cost = 0.20  ← Stuck in shallow minimum
Epoch 200: Cost = 0.25  ← Escapes!
Epoch 300: Cost = 0.08  ← Finds deeper minimum (better!)
```

**How to recognize:** Cost rises then drops significantly below previous minimum. Keep training!

## Cost Stuck / Not Decreasing

**"My cost is stuck at 0.2 and won't improve!"**

### Quick Fixes (Try in Order):

1. **Increase Learning Rate**
   ```python
   learning_rate=0.01  # Instead of 0.001
   ```
   Most common cause - learning rate too low.

2. **Train Longer**
   ```python
   epochs=2000  # Instead of 500
   ```
   Sometimes it just needs more time.

3. **Restart with Different Initialization**
   Just re-run the script. Random weights might have started in a bad spot.

4. **Relax Gradient Clipping**
   ```python
   clip_value=5  # Instead of 1
   ```
   Too tight clipping prevents large updates needed to escape plateaus.

5. **Check Data Normalization**
   ```python
   # RGB should be divided by 255:
   normalized = [r/255.0, g/255.0, b/255.0]
   ```

6. **Add More Neurons/Layers**
   Network might be too small to learn the pattern.

7. **Switch Cost Function**
   ```python
   cost_function='mae'  # Try MAE instead of MSE
   ```

8. **Check if 0.2 is Actually Good**
   - For binary classification: Cost ~0.5 = random, ~0.2 = decent, ~0.05 = very good
   - Check actual accuracy on test data!

## Understanding Cost vs Accuracy

**"My cost is 0.8 but accuracy is 96% - is this normal?"**

**Yes! Cost and accuracy measure different things:**

**Accuracy** = "Did you get the right class?" (binary: correct or wrong)
**Cost** = "How confident and correct are your predictions?" (continuous)

### Example:

For binary classification with sigmoid output and targets `[1, 0]`:

```python
# Barely correct prediction:
predicted = [0.6, 0.4]  # Correct! (0.6 > 0.4)
target = [1, 0]
cost = 0.5*(0.6-1)² + 0.5*(0.4-0)² = 0.16
accuracy = 100%  # Classified correctly!

# Very confident prediction:
predicted = [0.99, 0.01]  # Very confident!
target = [1, 0]
cost = 0.5*(0.99-1)² + 0.5*(0.01-0)² = 0.00005
accuracy = 100%  # Also classified correctly!
```

**What different costs mean:**

| Cost | Meaning | Example Prediction |
|------|---------|-------------------|
| ~0.0-0.1 | Very confident, correct | `[0.95, 0.05]` vs target `[1, 0]` |
| ~0.2-0.4 | Correct but low confidence | `[0.7, 0.3]` vs target `[1, 0]` |
| ~0.5-0.8 | Barely correct or wrong | `[0.55, 0.45]` vs target `[1, 0]` |
| ~1.0+ | Very wrong | `[0.2, 0.8]` vs target `[1, 0]` |

**High accuracy + high cost = network knows the answer but isn't confident**

Solution: Keep training to improve confidence (reduce cost).

## Negative Costs

**"My cost is -13.0 - how is that possible?!"**

**This means you're using Binary Cross-Entropy with [-1, 1] target data!**

Binary CE requires targets in [0, 1] range. See the [Binary Cross-Entropy section](#binary-cross-entropy) for details.

**Quick fix:**
```python
# Option 1: Change data format to [0, 1]
target_data = [[1, 0], [0, 1], ...]  # Instead of [[1, -1], [-1, 1], ...]

# Option 2: Use MSE/MAE instead
training = Training(net, ..., cost_function='mse')  # Works with [-1, 1] data
```

## Recommended Hyperparameters

Based on common usage in this repo:

### Learning Rate

| Scenario | Recommended LR |
|----------|---------------|
| **Initial training (simple problem)** | 0.001 to 0.01 |
| **Initial training (complex problem)** | 0.0001 to 0.001 |
| **Continue training (fine-tuning)** | 0.00005 to 0.0001 (10x lower) |
| **Cost exploding** | Divide current LR by 10 |
| **Cost stuck** | Multiply current LR by 10 |

### Gradient Clipping

| Scenario | Recommended clip_value |
|----------|----------------------|
| **Default / Most cases** | 5 |
| **Gradient explosion** | 1 |
| **Cost stuck at plateau** | 10 |

### Epochs

| Scenario | Recommended epochs |
|----------|-------------------|
| **Simple problems (XOR)** | 500-1000 |
| **Medium problems (RGB, Sine)** | 1000-3000 |
| **Complex problems (Checkerboard)** | 3000-5000 |
| **Continue training** | 300-500 |

**Pro tip:** Use checkpointing (enabled by default) and just set a high epoch count (5000+). Training will auto-save the best model even if it overfits later!
-----