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
 - [x] Add optimization so the network doesn't train too slowly on large datasets, like mini-batch training?
    - added `batch_size` parameter to Training constructor for mini-batch gradient descent<br>

 ❌ Allow setting a random seed for reproducibility when initing a network
    - decided against implementing this, as its trivial and i'm too lazy lol and when creating MLPs you will basically never ever use this<br>

 ❌ Adding noise to training data when processing (like MNIST) if the data is regularized, to be able to train more robust networks
    - decided against this, as that's for the user to process their own data lol

# Known Issues
If you set...
 - the learning rate too high (>0.001)
 - too high of a clipping barrier (haven't tested but its a given because when I didn't clip, it just killed itself)
 - or initialize layer weights to be too large (I initially did it with a normal distr mean0 and std1, but I had to lower the std)

the network will diverge due to vanishing or exploding gradients (see `/docs/GRADIENT_ANALYSIS.md` for additional details). This apparently is a common issue with neural networks, and is usually solved by clipping the network or by using a lower learning rate.

**He/Xavier initialization requires normalized data!** If you use `weight_init='he'` or `weight_init='xavier'` with unnormalized data (e.g., raw RGB 0-255), the large inputs × large weights = exploding activations and training fails. Either normalize your data first (divide RGB by 255.0 to get 0-1 range), or use a weight init like `weight_init='normal', weight_init_params={'std': 0.01}`.

If the overall dataset size is small, or if the data is regularized/preprocessed a certain way (like MNIST <i>cough cough</i>), the network may overfit. Consider adding noise/scaling/fucking around with the data, and using `samples_per_epoch` to train on a random subset each epoch for regularization. Or, if network trains too slowly per epoch, just use sampling as well.

The network initializes weights and biases randomly, so training results may vary between runs. So I guess you can just initialize multiple times and pick the best one, then continue training from there. You could also set a random seed for reproducibility if desired (I haven't implemented, but you can set `np.random.seed(your_seed)` at the start of your script). 

This is a pedagogical-ish codebase, so the training backpropogation processes one sample at a time, and thus isn't optimized. For the mini-batch training, we are accumulating gradients in a loop, instead of processing them at once for example in a matrix (an array of input arrays, instead of one input array at a time). You could refactor all the code to accept matrices and do backprop and forward prop with matrices, but it kinda defeats the purpose of showing stuff clearly imo. So do be advised the code isn't the most efficient possible implementation. 

# Notes
Numpy is the only dependency. The code is written noobishly so optimization would be nice.

An epoch is one iteration through the entire training set. 

See `/docs/` for some more specific and supplementary documentation.

Example models that showcase all features are made in `main_create_and_train.py`. Currently, I have 7 networks that showcase all features, each trained on different problems:
 - RGB color classification (is this color "red" or "not red"?)
 - XOR (+normalized noise) problem
 - Sine wave approximation
 - Checkerboard pattern classification
 - Quadrant classification (which quadrant does this 2D point belong to?)
 - Iris flower classification 
 - Linear regression (simple y=mx+b fitting, which apparently is a thing)

Some of these models are good, others are pretty bad, but they should serve as decent and comprehensive examples of how to use the code.

More optimized models I've made are in `main_optimized_mlps.py`, that have better architectures overall (since they're not showcasing features). I have 8 of them, each trained on different problems:
 - RGB color classification (is this color "red" or "not red"?)
 - XOR (+normalized noise) problem
 - Sine wave approximation
 - Checkerboard pattern classification
 - Quadrant classification (which quadrant does this 2D point belong to?)
 - Iris flower classification 
 - Linear regression (simple y=mx+b fitting, which apparently is a thing)
 - MNIST digit classification

The MNIST model in particular is a standard example of MLP usage, and this one actually trains decently well in my first attempt with ~94% accuracy after 500 epochs of initial training, and  ~96% accuracy after 200 epochs of fine-tuning.

The `main_create_and_train.py`, `main_load.py`, and the visualize scripts only reflect my most recent edits/training attempts, so be sure to check them and modify so you correctly train/visualize stuff you want.

Training everything lowkey is kinda finicky, you might need to restart multiple times to get a good initialization that trains well. Further refining of the network may need specific data generation. For example, if you want to refine the boundary of the sine categorization problem, you can generate more data that is clustered around the boundary en masse, and train with that, which would force the network to try to improve its boundary performance. 

I would recommend starting with pure SGD training to get to a good position first, then you can proceed with mini-batch training with a reasonable number of samples (16-256 ish) to fine tune, as mini-batch training typically is much more smooth and doesn't jump around too hard. Again, training with pure SGD alone works, but using mini-batch to fine tune is smoother and gives better training times and results. I did implement mini-batch training after i finalized a lot of the example neural nets, but I've tested other nets with it and it works properly, so use as needed without fear :)

MNIST data json files are too bulky and annoying, so I've excluded them from the repo in the gitignore. You can generate them yourself using `utility_mnist_processing.py`.

Also, even with best model checkpointing, I would recommend making a backup of the model before you run any continue training script, because the checkpointing will save the model after *one gradient step has already been done*, and does not save the original model. This way, if the model just purely gets worse, then you can revert to the original save. Otherwise, checkpointing does its job pretty well.

# Usage
Run the data generation scripts to get data. 

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

## Batch Training
You can add batch training to average gradients in a batch of size *n* by setting it in the training object. For example, you can just add `batch_size=config.get('batch_size', 1)` to the training object in `main_coninue_training.py` if you want to add mini-batch training to it, and add parameter `'batch_size': 32` to your model training parameters above lol.
```python
# Make a training object with a default batch size of 1 if not supplied, or whatever the user chose
training = Training(neural_net, learning_rate=config['learning_rate'], clip_value=config['clip_value'],
cost_function=config['cost_function'], checkpoint_path=model_file, batch_size=config.get('batch_size', 1))
```

## Automatic Best Model Checkpointing
To prevent loss of the best model due to overfitting or cost spikes, enable automatic checkpointing. The best model (lowest cost) is saved to a specified file whenever a new best cost is achieved during training. However, I guess you can argue grokking might be prevented by this, but idk if small models can even "grok" lol (also weight decay would need to be implemented :p).
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

## Make sure to also check out `/visualization/` :3

here's some images of my trained mlps 
<img src="/visualization/checkerboard_visualization.png" alt="Checkerboard Visualization"/>
<img src="/visualization/house_price_visualization.png" alt="House Price Visualization"/>
<img src="/visualization/mnist_visualization.png" alt="MNIST Visualization"/>
<img src="/visualization/quadrant_visualization.png" alt="Quadrant Visualization"/>
<img src="/visualization/red_color_visualization.png" alt="Red Color Visualization"/>
<img src="/visualization/sine_visualization.png" alt="Sine Wave Visualization"/>
<img src="/visualization/xor_visualization.png" alt="XOR Visualization"/>

here's html files that let you play with the trained mlps interactively<br>
download the raw file and open it in your browser to see :3
- [Iris Classifier](visualization/iris_visualization_4d.html)
- [Red Color Classifier](visualization/red_color_visualization_3d.html)

And lastly, there's an interactive MNIST classifier. You have to manually upload the `model_mnist.json`, and make sure you have the 4 font files in `./visualization/fonts/` to render the page properly, but you're able to draw on a canvas and see what the trained model predicts. Beware that since MNIST data is regularized (centered, size nomalized, etc), a model that trains on purely the database alone will overfit into that specific regularization, and probably won't entirely be up to expectations on accuracy when testing on your own handwritten digits. You can do some finicky things like adding noise, uncentering, scaling, &c. the dataset entries to make a more robust network, but I'm too lazy to do that (also training takes forever lol)
- [Interactive MNIST Classifier](visualization/mnist_interactive.html)
