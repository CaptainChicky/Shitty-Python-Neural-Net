# Shitty-Python-Neural-Net
read the title lmao

I don't plan to update this in the forseeable future. Pull requests/issues welcome.

# Todo
1. Make the someActivationFunction.derivative thing work instead of manually setting it in layers
2. Instead of MSE cost, use cross entropy
3. Perhaps instead of tanh as the output layer activation function, use softmax, or maybe even sigmoid
4. Allow the training to choose a certain subset of the total data to train with for a single epoch.
5. Other stuff in the commented TODOs in each part of the code files

(The latter three are standard for modern neural networks)

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