
# Cost Functions Guide

**Default:** MSE (Mean Squared Error), works with most use cases

**Available cost functions:**
- `'mse'` - Mean Squared Error
- `'mae'` - Mean Absolute Error
- `'binary_crossentropy'` - Binary Cross-Entropy (for binary classification)
- `'categorical_crossentropy'` - Categorical Cross-Entropy (for multi-class classification)

To use a different cost function, simply pass the desired cost function name as the `cost_function` parameter when creating the `Training` object:
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

### Mean Squared Error (MSE)
```python
training = Training(neural_net, learning_rate=0.001, clip_value=1, cost_function='mse')
```
- **Formula:** `0.5 * (predicted - target)²`
- **When to use:** General-purpose, works with most output activation and data ranges
- **Works with:** Tanh [-1, 1], Sigmoid [0, 1], or most other activations
- **Data format:** Any range (e.g., [-1, 1], [0, 1], &c.)
- **Pros:** Smooth gradients, penalizes large errors more heavily
- **Cons:** Sensitive to outliers

### Mean Absolute Error (MAE)
```python
training = Training(neural_net, learning_rate=0.001, clip_value=1, cost_function='mae')
```
- **Formula:** `|predicted - target|`
- **When to use:** When you want to be less sensitive to outliers than MSE
- **Works with:** Tanh [-1, 1], Sigmoid [0, 1], or most other activations
- **Data format:** Any range (e.g., [-1, 1], [0, 1], &c.)
- **Pros:** Less sensitive to outliers, more robust
- **Cons:** Gradient is not smooth at predicted = target

### Binary Cross-Entropy
```python
training = Training(neural_net, learning_rate=0.001, clip_value=1, cost_function='binary_crossentropy')
```
- **Formula:** `-[target * log(predicted) + (1-target) * log(1-predicted)]`
- **When to use:** Binary classification (two classes only)
- **Works with:** Sigmoid output activation [0, 1]
- **Data format:** Targets **MUST** be 0 or 1 (e.g., `[1, 0]` for class A, `[0, 1]` for class B)
- **Pros:** Standard for binary classification, works well with sigmoid
- **Cons:** Requires specific data format

**⚠️Binary Cross-Entropy Requires [0, 1] Target Data!**

### Categorical Cross-Entropy
```python
training = Training(neural_net, learning_rate=0.001, clip_value=1, cost_function='categorical_crossentropy')
```
- **Formula:** `-sum(target * log(predicted))`
- **When to use:** Multi-class classification (3+ classes)
- **Works with:** Softmax output activation (probability distribution)
- **Data format:** Targets should be one-hot encoded (e.g., `[1, 0, 0, 0]` for class 1)
- **Pros:** Standard for multi-class classification, works perfectly with softmax
- **Cons:** Requires one-hot encoded targets and softmax output

**⚠️Categorical Cross-Entropy Requires Softmax Output Activation!**

## Cost Function & Output Activation Compatibility

**Softmax** must be used on **output layers only** (as a hardcoded restriction). Softmax must also be used **only with cat CE** (as another hardcoded restriction).

All other activations are unrestricted and can be used on hidden or output layers. In particular, 
 - **Sigmoid/Tanh**: Commonly used on output for classification, rarely on hidden (vanishing gradients)
 - **ReLU/Leaky ReLU/ELU**: Commonly used on hidden layers, can be used on output  for regression
 - Other stuff are whatever lol use them at your discretion.
 - **Linear**: Primarily for regression output (unbounded), rarely on hidden (no non-linearity)

But technically all work on both hidden and output layers

### Recommended Combinations

This compatibility matrix is for **OUTPUT LAYER** activations only.

| Cost Function | Compatible **Output** Activations | Data Format | Use Case |
|---------------|----------------------|-------------|----------|
| **MSE** | Tanh, Sigmoid, ReLU, Leaky ReLU, ELU, Linear | Any range | General-purpose, regression |
| **MAE** | Tanh, Sigmoid, ReLU, Leaky ReLU, ELU, Linear | Any range | General-purpose, robust to outliers |
| **Binary Cross-Entropy** | **Sigmoid ONLY** | Targets: 0 or 1 | Binary classification (2 classes) |
| **Categorical Cross-Entropy** | **Softmax ONLY** | Targets: one-hot encoded | Multi-class classification (3+ classes) |

### Detailed Compatibility Matrix

| **Output** Activation | MSE | MAE | Binary CE | Categorical CE | Typical Use |
|------------|-----|-----|-----------|----------------|-------------|
| **Tanh** | ✅ Works | ✅ Works | ❌ **WRONG** (negative outputs clipped) | ❌ **WRONG** (negative outputs clipped) | Classification with [-1, 1] data |
| **Sigmoid** | ✅ Works | ✅ Works | ✅ **RECOMMENDED** | ⚠️ **WRONG** (sum ≠ 1, not a prob distr) | Binary classification |
| **Softmax** | ❌ **WRONG GRADIENTS** | ❌ **WRONG GRADIENTS** | ❌ **WRONG GRADIENTS** | ✅ **REQUIRED** | Multi-class classification |
| **Linear** | ✅ **RECOMMENDED** | ✅ **RECOMMENDED** | ❌ **WRONG** (unbounded, not [0, 1]) | ❌ **WRONG** (unbounded, not [0, 1]) | **Regression** (unbounded outputs) |
| **ReLU/Leaky ReLU/ELU** | ✅ Works | ✅ Works | ❌ **WRONG** (unbounded, not [0, 1]) | ❌ **WRONG** (unbounded, not [0, 1]) | Regression (but Linear is better) |

**Notes:**
- **Softmax**: Only activation with hardcoded restrictions - MUST be on output layer with Categorical CE
- **All others**: Can technically be used anywhere, but typical usage patterns vary (see above)
- Ones not mentioned in the table (like gelu, swish, etc) are uncommon combinations but you can test at discretion typically with MSE/MAE.
- Both cross-entropy cost functions include automatic clipping (`epsilon = 1e-15`) to prevent `log(0)` errors. This ensures stable training even with extreme predictions.

## Backpropagation Notes

### Softmax + Categorical Cross-Entropy (Hardcoded Optimization)

When you use **softmax output activation** with **categorical cross-entropy loss**, the backpropagation automatically detects this combination and uses a **hardcoded optimization**:

```
gradient = predicted_values - target_values
```

**Why this is hardcoded:**
- Softmax has a Jacobian matrix derivative (not element-wise like other activations)
- Cannot use standard element-wise multiplication: `(dCost/dPredicted) * (dPredicted/dZ)`
- The full Jacobian-vector product simplifies to: `predicted - target`
- This is a well-known result in deep learning and is numerically stable

### All Other Combinations (Element-wise)

For **all other activation + cost function combinations**, backpropagation uses the standard element-wise chain rule:

```
gradient = (dCost/dPredicted) * (dPredicted/dZ)
```

This works correctly for element-wise activations like:
- **Sigmoid** with any cost function, including bin CE, which naturally simplifies to `predicted - target` through the multiplication.
- **Tanh, ReLU, Leaky ReLU, ELU, etc** with any cost function save for CE ones.

## Switching Cost Functions During Training

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
- Saved models only contain weights, biases, and architecture, no cost function
- When you create a new `Training` object, you can choose any compatible cost function

**Restrictions:**
- The same activation/cost compatibility rules still apply (see Compatibility Matrix)
- ❌ Can't switch TO binary_crossentropy unless output is sigmoid
- ❌ Can't switch TO categorical_crossentropy unless output is softmax
- ❌ Can't switch FROM softmax to anything except categorical_crossentropy
- ✅ Can freely switch between MSE ⟷ MAE for any activation

**Practical recommendations:**
- ✅ **MSE ⟷ MAE are switchable** - Both are regression losses with different error metrics, switching makes sense
- ❌ **Binary CE and Categorical CE, pick one and stick with it** - These are classification losses with different problem framings, switching mid-training doesn't make practical sense

**Practical use case for MSE ⟷ MAE:**
1. **Start with MAE** (robust to outliers, gets "close enough")
2. **Switch to MSE** (fine-tunes, penalizes large errors more heavily)

This is a legitimate training strategy, as MAE is less sensitive to outliers early on, then MSE polishes the fit.