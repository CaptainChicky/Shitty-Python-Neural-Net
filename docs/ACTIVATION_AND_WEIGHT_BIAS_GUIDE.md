# Activation Functions Guide
- **Input layer:** No activation (just passes data through)
- **Hidden layers:** Leaky ReLU (default, alpha=0.01) - alternatives: ReLU, ELU
  - Note: Sigmoid/Tanh can be used but suffer from vanishing gradients
- **Output layer:** Tanh (default) - alternatives: Sigmoid (binary classification), Softmax (multi-class classification), Linear (regression)
 
## Available Activation Functions
 
### Sigmoid
- **Formula:** `f(x) = 1 / (1 + e^(-x))`
- **Range:** (0, 1)
- **When to use:** Output layer for binary classification, typically paired with bin CE loss
- **Issues:** Suffers from vanishing gradients, saturates easily
- **Usage:**
  ```python
  layer = Layer(10, 5, 'hidden', activation_func=ActivationFunction.sigmoid)
  ```
 
### Tanh (Hyperbolic Tangent)
- **Formula:** `f(x) = tanh(x)`
- **Range:** (-1, 1)
- **When to use:** Output layer typically, for binary classification or regression with [-1, 1] targets
- **Issues:** Still has vanishing gradient issues
- **Usage:** Automatically used for output layers, or:
  ```python
  layer = Layer(10, 5, 'hidden', activation_func=ActivationFunction.tanh)
  ```
 
### ReLU (Rectified Linear Unit)
- **Formula:** `f(x) = max(0, x)`
- **Range:** [0, ∞)
- **When to use:** Fast training, simple problems, when you don't care about "dying neurons"
- **Issues:** Neurons can get stuck at 0 and stop learning
- **Usage:**
  ```python
  layer = Layer(10, 5, 'hidden', activation_func=ActivationFunction.relu)
  ```
 
### Leaky ReLU
- **Formula:** `f(x) = max(alpha * x, x)` where `alpha=0.01` by default
- **Range:** (-∞, ∞)
- **When to use:** Almost always for hidden layers (it's the default!)
- **Issues:** Generally none, better than standard ReLU since it doesn't have dying neurons
- **Usage:**
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
  - `f(x) = alpha * (e^x - 1)` when x ≤ 0<br>
  where `alpha=1.0` by default
- **Range:** (-alpha, ∞)
- **When to use:** When you want smoother gradients than leaky ReLU, can help training converge faster
- **Issues:** Slightly more computationally expensive due to exp(), but smoother than leaky ReLU
- **Usage:**
  ```python
  # Default alpha=1.0
  layer = Layer(10, 5, 'hidden', activation_func=ActivationFunction.elu)
 
  # Custom alpha
  layer = Layer(10, 5, 'hidden',
                activation_func=ActivationFunction.elu,
                activation_params={'alpha': 0.5})
  ```

### Parametric ELU (Dual-Parameter)
- **Formula:**
  - `f(x) = alpha * x` when x > 0
  - `f(x) = beta * (e^x - 1)` when x ≤ 0<br>
  where `alpha=1.0` and `beta=1.0` by default
- **Range:** (-beta, ∞)
- **When to use:** Advanced tuning, combines Leaky ReLU (alpha) and ELU (beta) behaviors
- **Issues:** More parameters to tune, generally overkill for most problems
- **Usage:**
  ```python
  layer = Layer(10, 5, 'hidden',
                activation_func=ActivationFunction.parametric_elu,
                activation_params={'alpha': 1.5, 'beta': 1.0})
  ```
 
### Linear (Identity Function)
- **Formula:** `f(x) = x`
- **Range:** (-∞, ∞)
- **When to use:** Output layer for regression with unbounded targets (house prices, temps, stock prices, &c.)
- **Issues:** Generally none, outputs are unbounded for regression tasks
- **Usage:**
  ```python
  output_layer = Layer(5, 1, 'output', activation_func=ActivationFunction.linear)
  ```

### Swish / SiLU (Sigmoid Linear Unit)
- **Formula:** `f(x) = x * sigmoid(alpha * x)` where `alpha=1.0` by default
- **Range:** (-∞, ∞)
- **When to use:** At discretion, could outperform ReLU
- **Issues:** More computationally expensive, non-monotonic (can be good or bad)
- **Usage:**
  ```python
  # Standard Swish (alpha=1.0)
  layer = Layer(10, 5, 'hidden', activation_func=ActivationFunction.swish)
 
  # Adjust alpha for more/less ReLU-like behavior
  layer = Layer(10, 5, 'hidden',
                activation_func=ActivationFunction.swish,
                activation_params={'alpha': 2.0})
  ```

### GELU (Gaussian Error Linear Unit)
- **Formula:** `f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`
- **Range:** (-∞, ∞)
- **When to use:** For MLPs, honestly never, as these are for transformer models (BERT, GPT), NLP tasks, modern architectures, but you can try it experimentally
- **Issues:** Computationally expensive, but smooth and probabilistically motivated
- **Usage:**
  ```python
  layer = Layer(10, 5, 'hidden', activation_func=ActivationFunction.gelu)
  ```
 
### Mish
- **Formula:** `f(x) = x * tanh(ln(1 + e^x))`
- **Range:** (-∞, ∞)
- **When to use:** Deep networks where you want smooth, self-regularized activation
- **Issues:** Most computationally expensive, but can outperform ReLU and Swish
- **Usage:**
  ```python
  layer = Layer(10, 5, 'hidden', activation_func=ActivationFunction.mish)
  ```
 
### SELU (Scaled Exponential Linear Unit)
- **Formula:**
  - `f(x) = scale * x` when x > 0
  - `f(x) = scale * alpha * (e^x - 1)` when x ≤ 0  
  where `alpha=1.67326324` and `scale=1.05070098` by default
- **Range:** (-1.758, ∞)
- **When to use:** Never really, as it's for self-normalizing neural networks, but you can try it experimentally
- **Issues:** Requires `lecun_normal` weight initialization and normalized data to work properly
- **Usage:**
  ```python
  # Use paper's default values for self-normalization
  layer = Layer(10, 5, 'hidden', activation_func=ActivationFunction.selu)
  
  # Custom parameters (advanced, usually keep defaults)
  layer = Layer(10, 5, 'hidden',
                activation_func=ActivationFunction.selu,
                activation_params={'alpha': 1.67326324, 'scale': 1.05070098})
  ```
 
### Softmax (Multi-Class Output)
- **Formula:** `f(x)_i = e^(x_i) / Σ(e^(x_j))` for all `j`
- **Range:** (0, 1) with outputs summing to 1.0
- **When to use:** Output layer for multi-class classification (3+ classes)
- **Issues:** None generally, this is the standard for multi-class problems
- **Usage:**
  ```python
  # For n-class classification (e.g., iris species)
  output_layer = Layer(10, 3, 'output', activation_func=ActivationFunction.softmax)
  
  # Training REQUIRES categorical cross-entropy
  training = Training(neural_net,
                     learning_rate=0.001,
                     clip_value=5,
                     cost_function='categorical_crossentropy')  # Required!
  ```

### Sign and Step
- **Usage:** Don't. I included them for funny, but they are completely useless.
 
## Layer-Specific Recommendations
 
### Input Layers
```python
input_layer = Layer(3, 3, 'input')
```
This just passes data through unchanged, so activation function doesn't matter, and weights/biases are unused.
 
### Hidden Layers
```python
# Default
hidden_layer = Layer(10, 5, 'hidden')  # Uses Leaky ReLU alpha=0.01
 
# Custom activation
hidden_layer = Layer(10, 5, 'hidden', activation_func=ActivationFunction.elu)
 
# Custom parameters
hidden_layer = Layer(10, 5, 'hidden',
                     activation_func=ActivationFunction.leaky_relu,
                     activation_params={'alpha': 0.02})
```
For activation functions, **Leaky ReLU (default)** is recommended for most cases. Alternatives like **ReLU** or **ELU** can be used depending on your problem. Avoid **Sigmoid** or **Tanh** in hidden layers due to vanishing gradient issues.

### Output Layers
```python
# Default
output_layer = Layer(5, 2, 'output') # Uses Tanh by default

# Custom activation
output_layer = Layer(5, 2, 'output', activation_func=ActivationFunction.sigmoid)

# Override with Softmax for multi-class classification
output_layer = Layer(5, 4, 'output', activation_func=ActivationFunction.softmax)
```
For output, it defaults to **Tanh**, but you can override it with **Sigmoid**,  **Softmax** (for multi-class classification), or **Linear** (for regression with unbounded outputs).
 
# Weight/Bias Guide
Weight initialization determines the starting values of your neural network's weights and biases. Poor initialization can cause vanishing/exploding gradients or slow training.

By default, hidden layers use **He initialization** for weights and **zeros** for biases, which works well with Leaky ReLU activations.
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
- **Formula:** Sample from `N(0, std²)`, default `std=0.01` if not specified
- **When to use:** With unnormalized data (e.g., RGB 0-255), or when you need very small weights

### Xavier/Glorot Initialization
```python
layer = Layer(10, 5, 'hidden',
              weight_init='xavier',
              weight_init_params={})  # No params needed
```
- **Formula:** `std = sqrt(2 / (fan_in + fan_out))`
- **When to use:** Sigmoid or Tanh activations in hidden layers, requires normalized data

### He Initialization
```python
layer = Layer(10, 5, 'hidden',
              weight_init='he',
              weight_init_params={})
```
- **Formula:** `std = sqrt(2 / fan_in)`
- **When to use:** ReLU, Leaky ReLU, ELU activations (default is Leaky ReLU), requires normalized data

### Uniform Distribution
```python
layer = Layer(10, 5, 'hidden',
              weight_init='uniform',
              weight_init_params={'limit': 0.1})
```
- **Formula:** Sample uniformly from `[-limit, limit]`
- **When to use:** Rarely, we prefer normal distributions in practice

### Uniform Xavier
```python
layer = Layer(10, 5, 'hidden',
              weight_init='uniform_xavier',
              weight_init_params={})
```
- **Formula:** `limit = sqrt(6 / (fan_in + fan_out))`
- **When to use:** Uniform variant of Xavier initialization, use at discretion, requires normalized data

## Available Bias Initializers

### Zeros
```python
layer = Layer(10, 5, 'hidden',
              bias_init='zeros',
              bias_init_params={})
```
Most common approach, and default.

### Ones
```python
layer = Layer(10, 5, 'hidden',
              bias_init='ones',
              bias_init_params={})
```
Rarely used, but can be tried for specific cases.

### Constant
```python
layer = Layer(10, 5, 'hidden',
              bias_init='constant',
              bias_init_params={'value': 0.5})
```
Sometimes used in LSTM networks, but generally not needed in MLPs. Test at your discretion.

### Normal Distribution
```python
layer = Layer(10, 5, 'hidden',
              bias_init='normal',
              bias_init_params={'std': 0.01})
```
Rarely used because all zeros are better, but can be tried experimentally.

## Recommendations by Activation Function

| Activation Function | Best Weight Init | Data Requirement |
|---------------------|------------------|------------------|
| Leaky ReLU (default) | `he` (default) | Normalized (0-1 or mean=0, std=1) |
| ReLU | `he` | Normalized |
| ELU | `he` | Normalized |
| Sigmoid | `xavier` | Normalized |
| Tanh | `xavier` | Normalized |
| Any (unnormalized data) | `normal` with std=0.01 | No normalization needed |

**Important note:** He and Xavier initialization assume **normalized input data** (mean≈0, std≈1). Without normalization, activations can saturate, causing vanishing gradients and failed training.