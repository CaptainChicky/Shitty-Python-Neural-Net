import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src directory to path (goes up to root, then into src)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from neural_network import NeuralNet

# Load the trained model
neural_net = NeuralNet()
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_checkerboard.json')
neural_net.load(model_path)

# Create a grid of points to visualize
resolution = 500  # Higher = smoother visualization
x = np.linspace(0, 4, resolution)
y = np.linspace(0, 4, resolution)
X, Y = np.meshgrid(x, y)

# Predict for each point in the grid
predictions = np.zeros((resolution, resolution))
for i in range(resolution):
    for j in range(resolution):
        point = np.array([X[i, j], Y[i, j]])
        output = neural_net.forward_propagation(point)
        # output[0] > output[1] means black, otherwise white
        predictions[i, j] = 1 if output[0] > output[1] else -1

# Create the actual checkerboard pattern
actual = np.zeros((resolution, resolution))
for i in range(resolution):
    for j in range(resolution):
        grid_x = int(X[i, j] // 1.0)
        grid_y = int(Y[i, j] // 1.0)
        actual[i, j] = 1 if (grid_x + grid_y) % 2 == 0 else -1

# Plot side by side
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Model's prediction
im1 = ax1.imshow(predictions, extent=[0, 4, 0, 4], origin='lower', 
                 cmap='RdYlBu', vmin=-1, vmax=1)
ax1.set_title('Model Prediction', fontsize=16)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.grid(True, alpha=0.3, color='orange', linewidth=0.5)
plt.colorbar(im1, ax=ax1, label='Black (1) vs White (-1)')

# Actual checkerboard
im2 = ax2.imshow(actual, extent=[0, 4, 0, 4], origin='lower', 
                 cmap='RdYlBu', vmin=-1, vmax=1)
ax2.set_title('Actual Checkerboard', fontsize=16)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.grid(True, alpha=0.3, color='orange', linewidth=0.5)
plt.colorbar(im2, ax=ax2, label='Black (1) vs White (-1)')

# Errors - yellow/orange (easier on eyes than red)
difference = np.abs(predictions - actual)
im3 = ax3.imshow(difference, extent=[0, 4, 0, 4], origin='lower', 
                 cmap='YlOrRd', vmin=0, vmax=2)
ax3.set_title('Errors (Model vs Actual)', fontsize=16)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.grid(True, alpha=0.3)
plt.colorbar(im3, ax=ax3, label='Error magnitude')

plt.tight_layout()

# Save to visualization directory
output_path = os.path.join(os.path.dirname(__file__), 'checkerboard_visualization.png')
plt.savefig(output_path, dpi=150)
plt.show()

# Calculate accuracy
accuracy = np.mean(predictions == actual) * 100
print(f"\nModel Accuracy: {accuracy:.2f}%")
print(f"Errors: {np.sum(difference > 0)} out of {resolution * resolution} points")