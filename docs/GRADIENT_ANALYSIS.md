Setting high `std` value, like `std=1`, will cause your network to have vanishing gradients due to output layer saturation with tanh. Other times, it may cause exploding gradients. Three scenarios are detailed below.


### SCENARIO 1: std=0.01 with Leaky ReLU ✅
- **Layer 1 weighted inputs:** -6.21 to 2.23 
- **Layer 2 weighted inputs:** -0.053 to 0.014 
- **Output:** [0.000066, -0.00030]
- **Max gradient:** 0.078
- **Result:** Cost decreases properly, network learns

### SCENARIO 2: std=1.0 with Leaky ReLU ❌
- **Layer 1 weighted inputs:** -482 to 392 
- **Layer 2 weighted inputs:** -944 to 875
- **Output:** [-1.0, -1.0] (notice how tanh is saturated)
- **Max gradient:** 0.000000 (notice no gradient)
- **Result:** Network completely stuck, can't learn at all

### SCENARIO 3: std=0.1 with Leaky ReLU ⚠️
- **Layer 1 weighted inputs:** -47.8 to 34.4 
- **Layer 2 weighted inputs:** -4.36 to 1.77 
- **Output:** [0.0096, -0.0107] 
- **Max gradient:** 1.79
- **Result:** Notice that we have to clip, or else gradients explode

Generally speaking, if the output layer is saturating (tanh near -1 or 1), using leaky ReLU in hidden layers won't help as backpropogation fails at the output layer. By keeping `std` small, like `std=0.01`, tanh remains in the linear region and gradients flow properly.

Alternatives are using different weight initialization schemes and changing the output activation/loss function:

1. Use other weight initializations:
   - Xavier/Glorot initialization: `std = sqrt(2 / (n_in + n_out))`
   - He initialization: `std = sqrt(2 / n_in)`

2. Use softmax instead of tanh for output

3. Use cross-entropy loss instead of MSE
