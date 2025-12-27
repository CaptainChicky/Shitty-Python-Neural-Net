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

# Notes
Numpy is the only dependency. The code is written noobishly so optimization would be nice.

An epoch is one iteration through the entire training set. 

See `/docs/` for some more specific and supplementary documentation.

Currently, I have 7 networks, each trained on different problems:
 - RGB color classification (is this color "red" or "not red"?)
 - XOR (+normalized noise) problem
 - Sine wave approximation
 - Checkerboard pattern classification
 - Quadrant classification (which quadrant does this 2D point belong to?)
 - Iris flower classification 
 - Linear regression (simple y=mx+b fitting, which apparently is a thing)

Some of these models are good, others are pretty bad, but they should serve as decent and comprehensive examples of how to use the code.

# Usage
Run the main scripts lmao and change them as you'd like to make your own MLP neural net.

<img src="/docs/fluttershy-mlp.png" alt="Fluttershy MLP" width="200"/>

## Basic Training
```python
# Train on all data each epoch (default behavior)
training.train(input_data, target_data, epochs=500)
```

## Subset Training
Apparently subset training adds regularization through data sampling so it overfits less on small datasets. The cost would jump around tho compared to basic training.
```python
# Train on 400 randomly selected samples per epoch (out of some total)
training.train(input_data, target_data, epochs=500, samples_per_epoch=400)
```

## Automatic Best Model Checkpointing
To prevent loss of the best model due to overfitting or cost spikes, enable automatic checkpointing. The best model (lowest cost) is saved to a specified file whenever a new best cost is achieved during training. However, I guess you can argue grokking might be prevented by this, but idk if small models can even "grok" lol.
```python
# The best model (lowest cost) is automatically saved during training
training = Training(neural_net,
                   learning_rate=0.001,
                   clip_value=5,
                   cost_function='mse',
                   checkpoint_path='models/model_best.json')  # Auto-saves best here

training.train(input_data, target_data, epochs=5000)
```
If you disable checkpointing, you have to manually save your model at the end of training.
```python
training = Training(neural_net,
                   learning_rate=0.001,
                   clip_value=5,
                   cost_function='mse')
                   # No checkpoint_path specified

# Train the model
training.train(input_data, target_data, epochs=5000)

# Manually save the final model after training
neural_net.save('models/model_final.json')
```

## Please refer to the `/docs/` folder for more detailed documentation
1. [Activation Functions Documentation](docs/ACTIVATION_AND_WEIGHT_BIAS_GUIDE.md)
2. [Cost Functions Documentation](docs/COST_FUNCTION_GUIDE.md)
3. [Gradient Analysis Documentation](docs/GRADIENT_ANALYSIS.md)
4. [Training Guide Documentation](docs/TRAINING_GUIDE.md)