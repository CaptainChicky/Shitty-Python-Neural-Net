# Shitty-Python-Neural-Net
read the title lmao

I don't plan to update this in the forseeable future. Pull requests/issues welcome.

# Todo
1. ~~Make the someActivationFunction.derivative thing work instead of manually setting it in layers~~ ✅ DONE - now automatically detects and sets derivatives
2. Instead of MSE cost, use cross entropy
3. Perhaps instead of tanh as the output layer activation function, use softmax, or maybe even sigmoid
4. Allow the training to choose a certain subset of the total data to train with for a single epoch.
5. ~~Optimize double forward propagation in training~~ ✅ DONE - backprop now returns both gradients and predictions
6. Allow custom alpha values for leaky relu (currently hardcoded to 0.01)
 
(Items 2-4 are standard for modern neural networks)

# Known Issues
If you set the learning rate too high (>0.001), too high of a clipping barrier (haven't tested but its a given because when I didn't clip, it just killed itself), or initialize layer weights to be too large (I initially did it with a normal distr mean0 and std1, but I had to lower the std), this network will diverge due to exploding gradients. This is a common issue with neural networks, and is usually solved by clipping the network or by using a lower learning rate. It's quite odd that such a small network will diverge, especially when the gradient technically is already clipped at 1 due to the implementation of leaky relu, but whatever lmao.

Since the total possible training dataset is just 256<sup>3</sup> possible inputs, the network may overfit. This is what the todo #4 will fix, likely.

The neural network is undertrained for values near (0, 0, 0) and (255, 255, 255) so it will output incorrect answers. This is likely due to the way how I've generated my data, but whatever lmao.

# Usage
Run the main scripts lmao and change them as you'd like to make your own neural net.

# Notes
Numpy is the only dependency.

Check TODO comments on each python file for a more complete todo.

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
- **Hidden layers:** Leaky ReLU (default, alpha=0.01) - alternatives: ReLU, ELU, Sigmoid, Tanh
- **Output layers:** Tanh (hardcoded) - alternatives: Sigmoid, Softmax (not implemented yet)
 
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
output_layer = Layer(5, 2, 'output')  # Always uses Tanh regardless of what you specify
```
 
**Current behavior:**
- **Hardcoded to Tanh** for all output layers
- **Why:** Our training data uses outputs in [-1, 1] range (e.g., `[1, -1]` for red, `[-1, 1]` for not-red)
- **Parameters:** Automatically set to empty dict (tanh has no parameters)
 
**Future alternatives (not implemented yet):**
- **Sigmoid:** For binary classification with 0/1 outputs instead of -1/1
- **Softmax:** For multi-class classification with probability distributions (this is standard for modern networks)
  - Example: Instead of `[1, -1, -1, -1]` for quadrant 1, softmax would output `[0.97, 0.01, 0.01, 0.01]` (probabilities that sum to 1)
  - Would require changing cost function from MSE to cross-entropy loss (see TODO #2)
 
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