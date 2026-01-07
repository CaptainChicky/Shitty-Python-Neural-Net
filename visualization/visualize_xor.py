import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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

# Predict for each point in the grid - keep RAW confidence values
predictions_raw = np.zeros((resolution, resolution))
predictions_binary = np.zeros((resolution, resolution))

for i in range(resolution):
    for j in range(resolution):
        point = np.array([X[i, j], Y[i, j]])
        output = neural_net.forward_propagation(point)
        
        # Raw confidence (0 to 1 for sigmoid)
        predictions_raw[i, j] = output[0]
        
        # Binary for accuracy (threshold at 0.5)
        predictions_binary[i, j] = 1 if output[0] > 0.5 else 0

# Create the actual XOR pattern
actual = np.zeros((resolution, resolution))
for i in range(resolution):
    for j in range(resolution):
        x_binary = 1 if X[i, j] > 0.5 else 0
        y_binary = 1 if Y[i, j] > 0.5 else 0
        actual[i, j] = (x_binary + y_binary) % 2

# Calculate confidence-based error
# For sigmoid: distance from the correct output (0 or 1)
confidence_error = np.abs(predictions_raw - actual)

# Create custom colormap: darker red (XOR=0) -> white -> darker blue (XOR=1)
colors_list = ['#ff9999', '#ffffff', '#9999ff']  # Darker red -> white -> darker blue
n_bins = 256
cmap_name = 'xor_custom'
xor_cmap = LinearSegmentedColormap.from_list(cmap_name, colors_list, N=n_bins)

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))

# ============================================
# Plot 1: Model's Prediction (RAW CONFIDENCE)
# ============================================
im1 = ax1.imshow(predictions_raw, extent=[-0.3, 1.3, -0.3, 1.3], origin='lower', 
                 cmap=xor_cmap, vmin=0, vmax=1, aspect='equal')

# Overlay training points
ax1.scatter(xor_zero_samples[:, 0], xor_zero_samples[:, 1],
           c='red', s=15, alpha=0.5, edgecolors='darkred', linewidths=0.5, 
           label='XOR = 0', zorder=5)
ax1.scatter(xor_one_samples[:, 0], xor_one_samples[:, 1],
           c='blue', s=15, alpha=0.5, edgecolors='darkblue', linewidths=0.5,
           label='XOR = 1', zorder=5)

# Draw model's decision boundary (thinner and lighter like checkerboard)
ax1.contour(X, Y, predictions_raw, levels=[0.5], colors='black', linewidths=1.5, 
           alpha=0.4, zorder=10)

# Mark the 4 pure XOR corners with BIG stars
pure_corners = [(0, 0), (0, 1), (1, 0), (1, 1)]
pure_results = [0, 1, 1, 0]
for (cx, cy), result in zip(pure_corners, pure_results):
    color = 'blue' if result == 1 else 'red'
    ax1.scatter(cx, cy, c=color, s=250, marker='*', edgecolors='black', 
               linewidths=2, zorder=15)

# Add corner labels OUTSIDE the plot area (like original)
ax1.text(0, -0.22, '(0,0)\nXOR=0', fontsize=10, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.5, alpha=0.6), zorder=20)
ax1.text(0, 0.78, '(0,1)\nXOR=1', fontsize=10, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.5, alpha=0.6), zorder=20)
ax1.text(1, -0.22, '(1,0)\nXOR=1', fontsize=10, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.5, alpha=0.6), zorder=20)
ax1.text(1, 0.78, '(1,1)\nXOR=0', fontsize=10, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.5, alpha=0.6), zorder=20)

ax1.set_xlabel('X₁', fontsize=14, fontweight='bold')
ax1.set_ylabel('X₂', fontsize=14, fontweight='bold')
ax1.set_title('Model Prediction', fontsize=16, fontweight='bold')
ax1.legend(fontsize=10, loc='upper left', markerscale=2)
ax1.grid(True, alpha=0.4, linewidth=1)
ax1.set_xlim(-0.3, 1.3)
ax1.set_ylim(-0.3, 1.3)

cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Sigmoid Output', fontsize=11)

# ============================================
# Plot 2: Actual XOR Pattern
# ============================================
im2 = ax2.imshow(actual, extent=[-0.3, 1.3, -0.3, 1.3], origin='lower', 
                 cmap=xor_cmap, vmin=0, vmax=1, aspect='equal')

# Draw the TRUE XOR boundaries
ax2.plot([0.5, 0.5], [-0.3, 1.3], 'black', linewidth=2, linestyle='--', 
        label='Decision boundaries')
ax2.plot([-0.3, 1.3], [0.5, 0.5], 'black', linewidth=2, linestyle='--')

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
    ax2.scatter(cx, cy, c=color, s=250, marker='*', edgecolors='black', 
               linewidths=2, zorder=10)

ax2.set_xlabel('X₁', fontsize=14, fontweight='bold')
ax2.set_ylabel('X₂', fontsize=14, fontweight='bold')
ax2.set_title('Actual XOR Pattern', fontsize=16, fontweight='bold')
ax2.legend(fontsize=10, loc='upper left')
ax2.grid(True, alpha=0.4, linewidth=1)
ax2.set_xlim(-0.3, 1.3)
ax2.set_ylim(-0.3, 1.3)

# Add invisible colorbar to maintain consistent sizing with plots 1 and 3
from matplotlib import cm
dummy_cbar = plt.colorbar(cm.ScalarMappable(cmap=xor_cmap), ax=ax2, fraction=0.046, pad=0.04)
dummy_cbar.ax.set_visible(False)

# ============================================
# Plot 3: Error Map (CONFIDENCE-BASED)
# ============================================
im3 = ax3.imshow(confidence_error, extent=[-0.3, 1.3, -0.3, 1.3], origin='lower', 
                 cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')

# Model boundary
ax3.contour(X, Y, predictions_raw, levels=[0.5], colors='red', linewidths=2, 
           linestyles='--', label='Model boundary', zorder=8)

# True boundaries
ax3.plot([0.5, 0.5], [-0.3, 1.3], 'blue', linewidth=2, linestyle='--', 
        label='True boundaries', zorder=7)
ax3.plot([-0.3, 1.3], [0.5, 0.5], 'blue', linewidth=2, linestyle='--', zorder=7)

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

cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cbar3.set_label('Absolute Error', fontsize=11)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='yellow', label='Low Error'),
    Patch(facecolor='red', label='High Error'),
    plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Model'),
    plt.Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='True'),
]
ax3.legend(handles=legend_elements, fontsize=10, loc='upper left')

plt.suptitle('XOR Problem - The Classic Test for Hidden Layers\n' +
             'Red = [XOR=0] (same inputs) | Blue = [XOR=1] (different inputs) | Star = Pure corner', 
             fontsize=16, fontweight='bold', y=1.0)
plt.tight_layout()

# Save
output_path = os.path.join(os.path.dirname(__file__), 'xor_visualization.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()

# Calculate statistics
accuracy = np.mean(predictions_binary == actual) * 100
mean_confidence_distance = np.mean(np.abs(predictions_raw - 0.5))  # Distance from decision boundary
mean_error = np.mean(confidence_error)

print(f"\n{'='*60}")
print(f"Model Accuracy (Binary):     {accuracy:.2f}%")
print(f"Mean Confidence Distance:    {mean_confidence_distance:.3f} (distance from 0.5 boundary)")
print(f"Mean Prediction Error:       {mean_error:.3f} (lower = better)")
print(f"Binary Errors:               {np.sum(predictions_binary != actual):,} / {resolution * resolution:,} points")
print(f"{'='*60}")

# Confidence breakdown (distance from decision boundary)
very_confident = np.sum((predictions_raw < 0.1) | (predictions_raw > 0.9))
confident = np.sum(((predictions_raw >= 0.1) & (predictions_raw < 0.3)) | 
                   ((predictions_raw > 0.7) & (predictions_raw <= 0.9)))
uncertain = np.sum((predictions_raw >= 0.3) & (predictions_raw <= 0.7))

total_points = resolution * resolution
print(f"\nConfidence Breakdown:")
print(f"  Very Confident (out < 0.1 or > 0.9):  {very_confident:,} ({very_confident/total_points*100:.1f}%)")
print(f"  Confident (0.1-0.3 or 0.7-0.9):       {confident:,} ({confident/total_points*100:.1f}%)")
print(f"  Uncertain (0.3-0.7):                  {uncertain:,} ({uncertain/total_points*100:.1f}%)")
print(f"{'='*60}")

# Training accuracy
correct = 0
for i in range(len(train_inputs)):
    output = neural_net.forward_propagation(train_inputs[i])
    predicted_xor = 1 if output[0] > 0.5 else 0
    actual_xor = train_outputs[i][0]
    if predicted_xor == actual_xor:
        correct += 1

train_accuracy = (correct / len(train_inputs)) * 100
print(f"\nTraining Accuracy: {train_accuracy:.2f}%")

# Accuracy on the 4 pure corners
print(f"\nPure XOR corners:")
for (cx, cy), expected in zip(pure_corners, pure_results):
    output = neural_net.forward_propagation(np.array([cx, cy]))
    predicted = 1 if output[0] > 0.5 else 0
    confidence = output[0] if expected == 1 else (1 - output[0])
    status = "✅" if predicted == expected else "❌"
    print(f"  ({cx},{cy}): Expected {expected}, Got {predicted} (confidence: {confidence:.3f}) {status}")