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
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_optimized_xor.json')
neural_net.load(model_path)

# Load training data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'xor_data.json')
with open(data_path, 'r') as f:
    data = json.load(f)

train_inputs = np.array(data['Input_Values'])
train_outputs = np.array(data['Output_Values'])

# Separate XOR=0 and XOR=1 samples
xor_one_mask = np.array([output[0] == 1 for output in train_outputs])
xor_one_samples = train_inputs[xor_one_mask]
xor_zero_samples = train_inputs[~xor_one_mask]

# Create a grid of points to visualize
resolution = 400
x = np.linspace(-0.3, 1.3, resolution)
y = np.linspace(-0.3, 1.3, resolution)
X, Y = np.meshgrid(x, y)

# Predict for each point in the grid
predictions = np.zeros((resolution, resolution))
for i in range(resolution):
    for j in range(resolution):
        point = np.array([X[i, j], Y[i, j]])
        output = neural_net.forward_propagation(point)
        predictions[i, j] = 1 if output[0] > output[1] else 0

# Create the actual XOR pattern
actual = np.zeros((resolution, resolution))
for i in range(resolution):
    for j in range(resolution):
        x_binary = 1 if X[i, j] > 0.5 else 0
        y_binary = 1 if Y[i, j] > 0.5 else 0
        actual[i, j] = (x_binary + y_binary) % 2

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))

# ============================================
# Plot 1: Model's Prediction
# ============================================
im1 = ax1.contourf(X, Y, predictions, levels=[-0.5, 0.5, 1.5], 
                   colors=['#ffcccc', '#ccccff'], alpha=0.6)

# Overlay training points FIRST
ax1.scatter(xor_zero_samples[:, 0], xor_zero_samples[:, 1],
           c='red', s=15, alpha=0.5, edgecolors='darkred', linewidths=0.5, 
           label='XOR = 0', zorder=5)
ax1.scatter(xor_one_samples[:, 0], xor_one_samples[:, 1],
           c='blue', s=15, alpha=0.5, edgecolors='darkblue', linewidths=0.5,
           label='XOR = 1', zorder=5)

# Draw model's decision boundary AFTER points (higher zorder)
ax1.contour(X, Y, predictions, levels=[0.5], colors='red', linewidths=4, zorder=10)

# Mark the 4 pure XOR corners with BIG stars
pure_corners = [(0, 0), (0, 1), (1, 0), (1, 1)]
pure_results = [0, 1, 1, 0]
for (cx, cy), result in zip(pure_corners, pure_results):
    color = 'blue' if result == 1 else 'red'
    ax1.scatter(cx, cy, c=color, s=300, marker='*', edgecolors='black', 
               linewidths=2.5, zorder=15)

# Add corner labels OUTSIDE the plot area to avoid overlap
ax1.text(0, -0.22, '(0,0)\nXOR=0', fontsize=10, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.5))
ax1.text(0, 1.22, '(0,1)\nXOR=1', fontsize=10, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.5))
ax1.text(1, -0.22, '(1,0)\nXOR=1', fontsize=10, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.5))
ax1.text(1, 1.22, '(1,1)\nXOR=0', fontsize=10, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.5))

ax1.set_xlabel('X₁', fontsize=14, fontweight='bold')
ax1.set_ylabel('X₂', fontsize=14, fontweight='bold')
ax1.set_title('Model Prediction', fontsize=16, fontweight='bold')
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.4, linewidth=1)
ax1.set_xlim(-0.3, 1.3)
ax1.set_ylim(-0.3, 1.3)
ax1.set_aspect('equal')

# ============================================
# Plot 2: Actual XOR Pattern (SIMPLIFIED)
# ============================================
im2 = ax2.contourf(X, Y, actual, levels=[-0.5, 0.5, 1.5], 
                   colors=['#ffcccc', '#ccccff'], alpha=0.7)

# Draw the TRUE XOR boundaries (just the decision lines within bounds)
# Diagonal from bottom-left to top-right separating quadrants
ax2.plot([0.5, 0.5], [-0.3, 1.3], 'black', linewidth=3, linestyle='--', 
        label='Decision boundaries')  # Vertical at x=0.5
ax2.plot([-0.3, 1.3], [0.5, 0.5], 'black', linewidth=3, linestyle='--')  # Horizontal at y=0.5

# Mark the 4 quadrants with text
ax2.text(0.25, 0.25, 'XOR = 0\n(0,0)', fontsize=13, ha='center', va='center', 
        fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
ax2.text(0.25, 0.75, 'XOR = 1\n(0,1)', fontsize=13, ha='center', va='center',
        fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
ax2.text(0.75, 0.25, 'XOR = 1\n(1,0)', fontsize=13, ha='center', va='center',
        fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
ax2.text(0.75, 0.75, 'XOR = 0\n(1,1)', fontsize=13, ha='center', va='center',
        fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Big stars at corners
for (cx, cy), result in zip(pure_corners, pure_results):
    color = 'blue' if result == 1 else 'red'
    ax2.scatter(cx, cy, c=color, s=350, marker='*', edgecolors='black', 
               linewidths=3, zorder=10)

# Key insight text
ax2.text(0.5, -0.15, 'NOT LINEARLY SEPARABLE\n(Needs hidden layers!)', 
        fontsize=12, ha='center', fontweight='bold', style='italic',
        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.9, linewidth=2))

ax2.set_xlabel('X₁', fontsize=14, fontweight='bold')
ax2.set_ylabel('X₂', fontsize=14, fontweight='bold')
ax2.set_title('Actual XOR Pattern', fontsize=16, fontweight='bold')
ax2.legend(fontsize=11, loc='upper left')
ax2.grid(True, alpha=0.4, linewidth=1)
ax2.set_xlim(-0.3, 1.3)
ax2.set_ylim(-0.3, 1.3)
ax2.set_aspect('equal')

# ============================================
# Plot 3: Errors
# ============================================
difference = (predictions != actual).astype(float)
im3 = ax3.contourf(X, Y, difference, levels=[0, 0.5, 1], 
                   colors=['#90EE90', '#FF6B6B'], alpha=0.7)

# Model boundary
ax3.contour(X, Y, predictions, levels=[0.5], colors='red', linewidths=3, 
           linestyles='--', label='Model boundary', zorder=8)

# True boundaries
ax3.plot([0.5, 0.5], [-0.3, 1.3], 'blue', linewidth=2.5, linestyle='--', 
        label='True boundaries', zorder=7)
ax3.plot([-0.3, 1.3], [0.5, 0.5], 'blue', linewidth=2.5, linestyle='--', zorder=7)

# Stars at corners
for (cx, cy), result in zip(pure_corners, pure_results):
    color = 'blue' if result == 1 else 'red'
    ax3.scatter(cx, cy, c=color, s=250, marker='*', edgecolors='black', 
               linewidths=2, zorder=10)

ax3.set_xlabel('X₁', fontsize=14, fontweight='bold')
ax3.set_ylabel('X₂', fontsize=14, fontweight='bold')
ax3.set_title('Errors (Model vs Actual)', fontsize=16, fontweight='bold')
ax3.grid(True, alpha=0.4, linewidth=1)
ax3.set_xlim(-0.3, 1.3)
ax3.set_ylim(-0.3, 1.3)
ax3.set_aspect('equal')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#90EE90', label='Correct'),
    Patch(facecolor='#FF6B6B', label='Error'),
    plt.Line2D([0], [0], color='red', linewidth=3, linestyle='--', label='Model'),
    plt.Line2D([0], [0], color='blue', linewidth=2.5, linestyle='--', label='True'),
]
ax3.legend(handles=legend_elements, fontsize=11, loc='upper left')

plt.suptitle('XOR Problem - The Classic Test for Hidden Layers\n' +
             'Red = XOR=0 (same inputs) | Blue = XOR=1 (different inputs) | * = Pure corners', 
             fontsize=16, fontweight='bold')
plt.tight_layout()

# Save
output_path = os.path.join(os.path.dirname(__file__), 'xor_visualization.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()

# Calculate accuracy
accuracy = np.mean(predictions == actual) * 100
print(f"\nOverall Accuracy: {accuracy:.2f}%")

# Training accuracy
correct = 0
for i in range(len(train_inputs)):
    output = neural_net.forward_propagation(train_inputs[i])
    predicted_xor = 1 if output[0] > output[1] else 0
    actual_xor = 1 if train_outputs[i][0] == 1 else 0
    if predicted_xor == actual_xor:
        correct += 1

train_accuracy = (correct / len(train_inputs)) * 100
print(f"Training Accuracy: {train_accuracy:.2f}%")

# Accuracy on the 4 pure corners
print(f"\nPure XOR corners:")
for (cx, cy), expected in zip(pure_corners, pure_results):
    output = neural_net.forward_propagation(np.array([cx, cy]))
    predicted = 1 if output[0] > output[1] else 0
    status = "✅" if predicted == expected else "❌"
    print(f"  ({cx},{cy}): Expected {expected}, Got {predicted} {status}")