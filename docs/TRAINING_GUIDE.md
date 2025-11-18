# Training Guide: 5 Classification Problems

## Quick Start

1. Generate data for the problem you want:
   ```bash
   python definition_of_red.py        # RGB color classification
   python definition_of_xor.py        # XOR problem
   python definition_of_sine.py       # Sine wave classification
   python definition_of_checkerboard.py  # Checkerboard pattern
   python definition_of_quadrant.py   # Quadrant classification
   ```

2. Edit `main_create_and_train.py`:
   - **Comment out** the current configuration (Configuration 1 by default)
   - **Uncomment** the configuration you want to train

3. Train:
   ```bash
   python main_create_and_train.py
   ```

4. Models are saved to different files automatically:
   - RGB → `model_params.json`
   - XOR → `model_xor.json`
   - Sine → `model_sine.json`
   - Checkerboard → `model_checkerboard.json`
   - Quadrant → `model_quadrant.json`

---

## Problem Details

### 1. RGB Red Color Classification (Original)
**File:** `definition_of_red.py` → `color_data.json`

- **Task:** Is this RGB color "red"?
- **Input:** 3 values (R, G, B from 0-255)
- **Output:** 2 classes (red vs not-red)
- **Architecture:** 3 → 10 → 5 → 2
- **Learning Rate:** 0.00001
- **Epochs:** 500
- **Difficulty:** Medium (class balance was the main issue)

**Example:**
- Input: `[255, 0, 0]` (pure red) → Output: `[1, -1]` (red)
- Input: `[0, 255, 0]` (green) → Output: `[-1, 1]` (not red)

---

### 2. XOR Problem (Classic Test)
**File:** `definition_of_xor.py` → `xor_data.json`

- **Task:** Learn XOR function
- **Input:** 2 values (x1, x2 near 0 or 1 with noise)
- **Output:** 2 classes (0 vs 1)
- **Architecture:** 2 → 4 → 2 (simpler network)
- **Learning Rate:** 0.0001 (higher for simpler problem)
- **Epochs:** 500
- **Difficulty:** Easy (but impossible without hidden layers!)

**Why it's important:** This problem proved that neural networks NEED hidden layers. A simple perceptron (no hidden layers) cannot solve XOR.

**Example:**
- Input: `[0, 0]` → XOR = 0 → Output: `[-1, 1]`
- Input: `[0, 1]` → XOR = 1 → Output: `[1, -1]`
- Input: `[1, 0]` → XOR = 1 → Output: `[1, -1]`
- Input: `[1, 1]` → XOR = 0 → Output: `[-1, 1]`

---

### 3. Sine Wave Classification
**File:** `definition_of_sine.py` → `sine_data.json`

- **Task:** Is point (x, y) above or below y = sin(x)?
- **Input:** 2 values (x ∈ [0, 2π], y ∈ [-1.5, 1.5])
- **Output:** 2 classes (above vs below)
- **Architecture:** 2 → 12 → 8 → 2 (more neurons for periodic pattern)
- **Learning Rate:** 0.00005 (lower for smooth curves)
- **Epochs:** 1000 (more for complex pattern)
- **Difficulty:** Hard (periodic/oscillating boundary)

**Example:**
- Input: `[π/2, 1.5]` → y > sin(π/2)=1 → Above → Output: `[1, -1]`
- Input: `[0, -0.5]` → y < sin(0)=0 → Below → Output: `[-1, 1]`

---

### 4. Checkerboard Pattern
**File:** `definition_of_checkerboard.py` → `checkerboard_data.json`

- **Task:** Is point on a black or white checkerboard square?
- **Input:** 2 values (x, y ∈ [0, 4])
- **Output:** 2 classes (black vs white)
- **Architecture:** 2 → 16 → 8 → 2 (many neurons for multiple boundaries)
- **Learning Rate:** 0.0001
- **Epochs:** 800
- **Difficulty:** Hard (many decision boundaries)

**Pattern:** (floor(x) + floor(y)) % 2 == 0 → black

**Example:**
- Input: `[0.5, 0.5]` → grid (0, 0) → 0+0=0 (even) → Black → Output: `[1, -1]`
- Input: `[0.5, 1.5]` → grid (0, 1) → 0+1=1 (odd) → White → Output: `[-1, 1]`
- Input: `[1.5, 1.5]` → grid (1, 1) → 1+1=2 (even) → Black → Output: `[1, -1]`

---

### 5. Quadrant Classification (MULTI-CLASS!)
**File:** `definition_of_quadrant.py` → `quadrant_data.json`

- **Task:** Which quadrant is point (x, y) in?
- **Input:** 2 values (x, y ∈ [-5, 5])
- **Output:** 4 classes (Q1, Q2, Q3, Q4) - **ONE-HOT ENCODING**
- **Architecture:** 2 → 8 → 6 → **4** (4 output neurons!)
- **Learning Rate:** 0.0001
- **Epochs:** 500
- **Difficulty:** Medium (but different! Multi-class instead of binary)

**This is special:** First multi-class problem! Output has 4 neurons instead of 2.

**Output Encoding:**
- Q1 (x>0, y>0): `[1, -1, -1, -1]`
- Q2 (x<0, y>0): `[-1, 1, -1, -1]`
- Q3 (x<0, y<0): `[-1, -1, 1, -1]`
- Q4 (x>0, y<0): `[-1, -1, -1, 1]`

**Example:**
- Input: `[3, 4]` → Quadrant 1 → Output: `[1, -1, -1, -1]`
- Input: `[-2, 1]` → Quadrant 2 → Output: `[-1, 1, -1, -1]`

---

## How to Switch Problems

### In `main_create_and_train.py`:

**Currently active:**
```python
# CONFIGURATION 1: RGB Red Color Classification
neural_net = NeuralNet()
input_layer = Layer(...)
# ... rest of config
```

**To switch to XOR:**
1. Comment out Configuration 1 (add `#` at start of each line)
2. Uncomment Configuration 2 (remove `#` from each line)

```python
############################################################################################################
# # CONFIGURATION 1: RGB Red Color Classification  (NOW COMMENTED OUT)
# ############################################################################################################
# neural_net = NeuralNet()
# ...

############################################################################################################
# CONFIGURATION 2: XOR Problem  (NOW ACTIVE)
############################################################################################################
neural_net = NeuralNet()
input_layer = Layer(previousLayer_size=2, layer_size=2, layer_type='input')
# ... rest of config
```

---

## Testing/Evaluating Models

After training, use `main_load.py` to evaluate your model:

1. Edit `main_load.py`:
   - Comment out the current configuration
   - Uncomment the configuration matching what you just trained

2. Run:
   ```bash
   python main_load.py
   ```

3. You'll get detailed metrics:
   - **Accuracy:** Overall % correct
   - **Confusion Matrix:** Shows which classes get confused
   - **Per-Class Precision:** Of predicted class X, % that were actually class X
   - **Per-Class Recall:** Of actual class X, % that were correctly identified
   - **F1-Score:** Harmonic mean of precision and recall

**Example Output:**
```
======================================================================
TESTING: RGB Red Color Classification
======================================================================

Loaded model from model_params.json
Loaded 500 test samples from color_data.json

======================================================================
RESULTS
======================================================================

Accuracy: 97.60%
Correct: 488/500

Confusion Matrix (rows=actual, cols=predicted):

Actual \ Pred  |       Red           Not Red
------------------------------------------------
      Red      |       250              0
    Not Red    |       12              238

Per-Class Metrics:
----------------------------------------------------------------------
Class                Precision    Recall       F1-Score
----------------------------------------------------------------------
Red                   95.42%      100.00%       97.66%
Not Red              100.00%       95.20%       97.54%

======================================================================
Testing complete!
======================================================================
```

---

## Expected Results

All problems should achieve >95% accuracy with balanced data:

- **XOR:** ~99-100% (simple pattern)
- **RGB:** ~97% (tested)
- **Sine:** ~95-98% (smooth boundary)
- **Checkerboard:** ~90-95% (complex boundaries)
- **Quadrant:** ~99-100% (simple linear boundaries, but 4 classes)

If you're getting 0% precision/recall, you probably forgot to generate the data file!

---

## Troubleshooting

**Problem:** `FileNotFoundError: 'xor_data.json'`
**Solution:** Run `python definition_of_xor.py` first

**Problem:** Wrong architecture error
**Solution:** Make sure you uncommented the right configuration

**Problem:** 0% precision/recall
**Solution:** Check if data file exists and has balanced classes

**Problem:** Cost not decreasing
**Solution:** Try different learning rate (increase if too slow, decrease if unstable)

**Problem:** Network outputs same value for everything
**Solution:** Check for class imbalance in data, regenerate with balanced dataset
