# Exploding Gradients Investigation Results

## TL;DR: It's NOT exploding gradients - it's VANISHING gradients!

The problem with std=1 initialization is **vanishing gradients caused by tanh saturation in the output layer**, not exploding gradients.

---

## What's Happening:

### SCENARIO 1: std=0.01 with Leaky ReLU ✅ (WORKS)
- **Layer 1 weighted inputs:** -6.21 to 2.23 (reasonable)
- **Layer 2 weighted inputs:** -0.053 to 0.014 (small)
- **Output:** [0.000066, -0.00030] (small, not saturated)
- **Max gradient:** 0.078 (reasonable)
- **Result:** Cost decreases properly, network learns

### SCENARIO 2: std=1.0 with Leaky ReLU ❌ (FAILS)
- **Layer 1 weighted inputs:** -482 to 392 (HUGE!)
- **Layer 2 weighted inputs:** -944 to 875 (MASSIVE!)
- **Output:** [-1.0, -1.0] (tanh SATURATED!)
- **Max gradient:** 0.000000 (VANISHED!)
- **Result:** Network completely stuck, can't learn at all

### SCENARIO 3: std=1.0 with ReLU ❌ (FAILS TOO!)
- **Layer 1 weighted inputs:** -335 to 573 (large)
- **Layer 2 weighted inputs:** -511 to 1250 (very large)
- **Output:** [1.0, -0.99999947] (tanh SATURATED!)
- **Max gradient:** 0.002666 (tiny)
- **Result:** ReLU doesn't help because the output layer still uses tanh

### SCENARIO 4: std=0.1 with Leaky ReLU ⚠️ (BORDERLINE)
- **Layer 1 weighted inputs:** -47.8 to 34.4 (manageable)
- **Layer 2 weighted inputs:** -4.36 to 1.77 (reasonable)
- **Output:** [0.0096, -0.0107] (small, not saturated)
- **Max gradient:** 1.79 (larger, gets clipped to 1)
- **Result:** Works but gradients are being clipped

---

## Root Cause Analysis:

### Why std=1 causes problems:

1. **Large weights** (std=1) → **Large weighted inputs**
   - With 3 inputs and 10 hidden neurons, each weighted input sums ~3 terms with std=1
   - This causes weighted inputs in the range of hundreds

2. **Large weighted inputs** → **Tanh saturation**
   - tanh(x) ≈ 1 when x > 3
   - tanh(x) ≈ -1 when x < -3
   - Your outputs are seeing inputs of -18 to -19 (SCENARIO 2)
   - This causes tanh to output exactly -1.0

3. **Tanh saturation** → **Zero derivatives**
   - tanh'(x) = 1/cosh²(x) ≈ 0 when |x| > 3
   - When tanh is saturated, its derivative is essentially 0
   - This causes the gradient to vanish during backpropagation

4. **Zero gradients** → **No learning**
   - With zero gradients, weight updates are zero
   - The network gets stuck and can't learn

### Why ReLU doesn't help:

- ReLU in **hidden layers** doesn't solve the problem because:
  - The **output layer** still uses tanh (hardcoded)
  - Large activations from ReLU layers still cause tanh saturation in the output
  - The vanishing gradient problem originates from the saturated output layer

### Why std=0.01 works:

- Small weights keep activations small
- Small activations keep tanh in its linear region (|x| < 1)
- In the linear region, tanh'(x) ≈ 1, so gradients flow properly
- The network can learn

---

## The Real Problem:

You wrote in your README that you have "exploding gradients" but the actual problem is:

**VANISHING GRADIENTS DUE TO OUTPUT LAYER SATURATION**

This is why gradient clipping doesn't help with std=1 - the gradients are already zero before clipping!

---

## Solutions:

### Current workaround (what you're doing):
✅ Use std=0.01 for weight initialization
- Keeps activations small
- Prevents tanh saturation
- Not optimal but works

### Better solutions (for future):

1. **Use proper weight initialization:**
   - Xavier/Glorot initialization: `std = sqrt(2 / (n_in + n_out))`
   - He initialization: `std = sqrt(2 / n_in)` (better for ReLU)
   - These are designed to keep activations in a reasonable range

2. **Use softmax instead of tanh for output:**
   - Softmax doesn't saturate as badly
   - Better for classification tasks
   - This is in your TODO list!

3. **Use cross-entropy loss instead of MSE:**
   - Cross-entropy with softmax has better gradient properties
   - Won't vanish as easily
   - Also in your TODO list!

4. **Add batch normalization:**
   - Normalizes activations between layers
   - Prevents saturation
   - More advanced technique

---

## Conclusion:

Your neural network doesn't have exploding gradients - it has **vanishing gradients** when you use std=1 because:

1. Large initial weights → Large activations
2. Large activations → Tanh saturation in output layer
3. Tanh saturation → Zero derivatives
4. Zero derivatives → Vanishing gradients → No learning

Your current solution (std=0.01) works but is suboptimal. Implementing proper weight initialization (Xavier/He) and switching from tanh+MSE to softmax+cross-entropy would be much better.

**The backpropagation logic is correct** - the issue is purely with initialization and activation function choice.
