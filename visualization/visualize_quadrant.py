import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from neural_network import NeuralNet

# Load the trained model
neural_net = NeuralNet()
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_quadrant.json')
neural_net.load(model_path)

# Load training data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'quadrant_data.json')
with open(data_path, 'r') as f:
    data = json.load(f)

train_inputs = np.array(data['Input_Values'])
train_outputs = np.array(data['Output_Values'])

# Convert one-hot encoding to class labels (0-3)
train_classes = np.argmax(train_outputs, axis=1)

# Create a grid of points to visualize
resolution = 500
x = np.linspace(-5, 5, resolution)
y = np.linspace(-5, 5, resolution)
X, Y = np.meshgrid(x, y)

# Predict for each point in the grid
predictions = np.zeros((resolution, resolution))
for i in range(resolution):
    for j in range(resolution):
        point = np.array([X[i, j], Y[i, j]])
        output = neural_net.forward_propagation(point)
        # Get the quadrant with highest activation
        predictions[i, j] = np.argmax(output)

# Create the actual quadrant pattern
actual = np.zeros((resolution, resolution))
for i in range(resolution):
    for j in range(resolution):
        x_val, y_val = X[i, j], Y[i, j]
        if x_val > 0 and y_val > 0:
            actual[i, j] = 0  # Q1
        elif x_val < 0 and y_val > 0:
            actual[i, j] = 1  # Q2
        elif x_val < 0 and y_val < 0:
            actual[i, j] = 2  # Q3
        else:
            actual[i, j] = 3  # Q4

# Define colors for each quadrant
colors = ['#ff9999', '#99ff99', '#9999ff', '#ffff99']  # Red, Green, Blue, Yellow
quadrant_names = ['Q1 (x>0, y>0)', 'Q2 (x<0, y>0)', 'Q3 (x<0, y<0)', 'Q4 (x>0, y<0)']

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# ============================================
# Plot 1: Model's Prediction
# ============================================
im1 = ax1.contourf(X, Y, predictions, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], 
                   colors=colors, alpha=0.6)

# Overlay training points
for q in range(4):
    mask = train_classes == q
    ax1.scatter(train_inputs[mask, 0], train_inputs[mask, 1],
               c=colors[q], label=quadrant_names[q], s=5, alpha=0.5, edgecolors='black', linewidths=0.3)

ax1.set_xlabel('X', fontsize=14)
ax1.set_ylabel('Y', fontsize=14)
ax1.set_title('Model Prediction', fontsize=16, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.2)
ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)

# ============================================
# Plot 2: Actual Quadrants
# ============================================
im2 = ax2.contourf(X, Y, actual, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], 
                   colors=colors, alpha=0.6)

# Add quadrant labels
ax2.text(2.5, 2.5, 'Q1\n(x>0, y>0)', fontsize=14, ha='center', va='center', fontweight='bold')
ax2.text(-2.5, 2.5, 'Q2\n(x<0, y>0)', fontsize=14, ha='center', va='center', fontweight='bold')
ax2.text(-2.5, -2.5, 'Q3\n(x<0, y<0)', fontsize=14, ha='center', va='center', fontweight='bold')
ax2.text(2.5, -2.5, 'Q4\n(x>0, y<0)', fontsize=14, ha='center', va='center', fontweight='bold')

ax2.set_xlabel('X', fontsize=14)
ax2.set_ylabel('Y', fontsize=14)
ax2.set_title('Actual Quadrants', fontsize=16, fontweight='bold')
ax2.grid(True, alpha=0.2)
ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)

# ============================================
# Plot 3: Errors
# ============================================
difference = (predictions != actual).astype(float)
im3 = ax3.contourf(X, Y, difference, levels=[0, 0.5, 1], 
                   colors=['#90EE90', '#FF6B6B'], alpha=0.7)  # Light green = correct, Light red = error

ax3.set_xlabel('X', fontsize=14)
ax3.set_ylabel('Y', fontsize=14)
ax3.set_title('Errors (Model vs Actual)', fontsize=16, fontweight='bold')
ax3.grid(True, alpha=0.2)
ax3.set_xlim(-5, 5)
ax3.set_ylim(-5, 5)

# Add legend for error plot
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#90EE90', label='Correct'),
                  Patch(facecolor='#FF6B6B', label='Error')]
ax3.legend(handles=legend_elements, fontsize=12, loc='upper right')

plt.suptitle('Quadrant Classification (4-Class Multi-Class Problem)', 
             fontsize=18, fontweight='bold')
plt.tight_layout()

# Save
output_path = os.path.join(os.path.dirname(__file__), 'quadrant_visualization.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()

# Calculate accuracy
accuracy = np.mean(predictions == actual) * 100
print(f"\nModel Accuracy: {accuracy:.2f}%")
print(f"Errors: {np.sum(difference)} out of {resolution * resolution} points")

# Per-quadrant accuracy
for q in range(4):
    mask_pred = predictions == q
    mask_actual = actual == q
    correct = np.sum((predictions == q) & (actual == q))
    total = np.sum(actual == q)
    acc = (correct / total) * 100 if total > 0 else 0
    print(f"{quadrant_names[q]}: {acc:.2f}% ({correct}/{total} points)")