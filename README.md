# Shitty-Python-Neural-Net
read the title lmao

I don't plan to update this in the forseeable future. Pull requests/issues welcome.

# Todo
1. Make the someActivationFunction.derivative thing work instead of manually setting it in layers
2. Instead of MSE cost, use cross entropy
3. Perhaps instead of tanh as the output layer activation function, use softmax
4. Allow the training to choose a certain subset of the total data to train with for a single epoch.
5. Other stuff in the commented TODOs in each part of the code files

(The latter three are standard for modern neural networks)

# Known Issues
If you set the learning rate too high (>0.001), too high of a clipping barrier (haven't tested but its a given because when I didn't clip, it just killed itself), or initialize layer weights to be too large (I initially did it with a normal distr mean0 and std1, but I had to lower the std), this network will diverge due to exploding gradients. This is a common issue with neural networks, and is usually solved by clipping the network or by using a lower learning rate. It's quite odd that such a small network will diverge but whatever lmao.

Since the total possible training dataset is just 256<sup>3</sup> possible inputs, the network may overfit. This is what the todo #4 will fix, likely.

The neural network is undertrained for values near (0, 0, 0) and (255, 255, 255) so it will output incorrect answers. This is likely due to the way how I've generated my data, but whatever lmao.

# Usage
Run the main scripts lmao. 

# Notes
Numpy is the only dependency.

Check TODO comments on each python file for a more complete todo.

An epoch is one iteration through the entire training set. You may generate new training sets but I haven't tested the net much with this thing because it seems to get stuck at some fixed cost and refuse to make any further progress lmao.

Currently, the network is being trained to recongnize if a given RGB color triple is "red" or not, as defined in my definition of red python script which also generates the training data. The code is written noobishly so optimization would be nice.

# Structure
Activation functions contain all activation functions and derivatives.

Definition of red contains the definition of what I defined as red, and generates training data.

Layers contain the implementation of the layer object, which contains the weights, biases, and activation function of a layer.

Neural Network contains the implementation of the neural network object, which contains the layers and loading/saving.

Training contains the actual functions that modify and train the network.

Main scripts are what you can run/train/blah the network with.

Training data is stored in the color_data.json, and the model is saved in the model_params.json file.
