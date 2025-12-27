## How to Start
1. Generate data for the problem you want or create/import your own data.

2. Edit `main_create_and_train.py` with the corresponding network number you want to train, or create your own network.

3. Run the file to train.

4. Save your model, or if you use checkpointing, it is auto-saved.

## Continued Training
To continue training an existing model, edit `main_create_and_train.py` and run the file.

## Testing/Evaluating Models
After training, use `main_load.py` to evaluate your model:

1. Edit `main_load.py` to evaluate your desired network/problem.

2. Run the file to evaluate.

3. You'll get detailed metrics:
   - **Accuracy:** Overall % correct
   - **Confusion Matrix:** Shows which classes get confused
   - **Per-Class Precision:** Of predicted class X, % that were actually class X
   - **Per-Class Recall:** Of actual class X, % that were correctly identified
   - **F1-Score:** Harmonic mean of precision and recall

## Troubleshooting

### My cost is increasing during training

Your earning rate is too high, and gradient descent takes steps that are too large and overshoots the minimum

You are overfitting to the training data. If you enabled checkpointing, the best model is still saved. Grokking may also occur if you train for a long time, but I wouldn't bet on that happenning lol.

When using `samples_per_epoch`, each epoch trains on different random samples, so cost may go up and down. Look at the overall trend over many epochs.

### My cost is not decreasing / stuck at a plateau

Try increasing learning rate, as you may be in a flat region of the cost landscape, or just keep training longer.

You can try restarting with a different initialization, as random weights might have started in a bad spot (just re-run the script).

You could also try relaxing the gradient clipping, as too tight clipping could prevent large updates needed to escape plateaus.

## Cost vs Accuracy

**Accuracy** = "Did you get the right class?"<br>
**Cost** = "How confident and correct are your predictions?"

For binary classification with sigmoid output and targets `[1, 0]`, an example is:

```python
# Unconfident but correct prediction:
predicted = [0.6, 0.4]  # Correct! (0.6 > 0.4)
target = [1, 0]
cost = 0.5*(0.6-1)² + 0.5*(0.4-0)² = 0.16
accuracy = 100%  # Classified correctly!

# Confident but wrong prediction:
predicted = [0.2, 0.8]  # Wrong! (0.2 < 0.8)
target = [1, 0]
cost = 0.5*(0.2-1)² + 0.5*(0.8-0)² = 0.325
accuracy = 0%  # Classified incorrectly!
```

If your cost ever becomes negative, this means you're using a wrong cost function for your data format. For example, using bin CE with [-1, 1] target data.

## Recommended Parameters

### Learning Rate

| Scenario | Recommended LR |
|----------|---------------|
| **Initial training (simple problem)** | 0.001 to 0.01 |
| **Initial training (complex problem)** | 0.0001 to 0.001 |
| **Continue training (fine-tuning)** | 0.00005 to 0.0001 (10x lower) |
| **Cost exploding** | Divide current LR by 10 |
| **Cost stuck** | Multiply current LR by 10 |

### Gradient Clipping

| Scenario | Recommended clip_value |
|----------|----------------------|
| **Default / Most cases** | 5 |
| **Cost stuck at plateau** | 10 |

### Epochs

| Scenario | Recommended epochs |
|----------|-------------------|
| **Simple problems (XOR)** | 500-1000 |
| **Medium problems (RGB, Sine)** | 1000-3000 |
| **Complex problems (Checkerboard)** | 3000-5000 |
| **Continue training** | 500-1000 |

tbh though you can use checkpointing and just set a high epoch count (5000+). Training will auto-save the best model even if it overfits later.

