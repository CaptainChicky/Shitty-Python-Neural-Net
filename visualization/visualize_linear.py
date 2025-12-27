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
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_linear_regression.json')
neural_net.load(model_path)

# Load training data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'linear_regression_data.json')
with open(data_path, 'r') as f:
    data = json.load(f)

test_inputs = np.array(data['Input_Values'])
test_outputs = np.array(data['Output_Values']).flatten()

# Get predictions
predictions = []
for i in range(len(test_inputs)):
    output = neural_net.forward_propagation(test_inputs[i])
    predictions.append(output[0])
predictions = np.array(predictions)

# Calculate metrics
mse = np.mean((predictions - test_outputs) ** 2)
mae = np.mean(np.abs(predictions - test_outputs))
rmse = np.sqrt(mse)
r2 = 1 - (np.sum((test_outputs - predictions) ** 2) / np.sum((test_outputs - np.mean(test_outputs)) ** 2))

print(f"\nModel Performance:")
print(f"  MAE:  ${mae:.2f}k")
print(f"  RMSE: ${rmse:.2f}k")
print(f"  R²:   {r2:.4f}")

# Create visualization with 5 subplots
fig = plt.figure(figsize=(20, 12))

feature_names = ['Square Footage', 'Bedrooms', 'Age']

# ============================================
# Plot 1: Predicted vs Actual (main plot)
# ============================================
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(test_outputs, predictions, alpha=0.5, s=20)
ax1.plot([test_outputs.min(), test_outputs.max()], 
         [test_outputs.min(), test_outputs.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Price ($1000s)', fontsize=12)
ax1.set_ylabel('Predicted Price ($1000s)', fontsize=12)
ax1.set_title('Predicted vs Actual House Prices', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.text(0.05, 0.95, f'R² = {r2:.4f}\nMAE = ${mae:.2f}k\nRMSE = ${rmse:.2f}k',
         transform=ax1.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================
# Plot 2: Residuals (errors)
# ============================================
ax2 = plt.subplot(2, 3, 2)
residuals = predictions - test_outputs
ax2.scatter(predictions, residuals, alpha=0.5, s=20)
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Predicted Price ($1000s)', fontsize=12)
ax2.set_ylabel('Residual (Predicted - Actual)', fontsize=12)
ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# ============================================
# Plot 3: Residual Distribution
# ============================================
ax3 = plt.subplot(2, 3, 3)
ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
ax3.axvline(x=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Residual ($1000s)', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Distribution of Errors', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# ============================================
# Plots 4-6: Individual Feature Effects
# ============================================
# For each feature, show how price changes with that feature (others at median)
median_features = np.median(test_inputs, axis=0)

for feat_idx in range(3):
    ax = plt.subplot(2, 3, 4 + feat_idx)
    
    # Create range for this feature
    feature_range = np.linspace(0, 1, 100)
    predicted_prices = []
    
    for val in feature_range:
        input_point = median_features.copy()
        input_point[feat_idx] = val
        pred = neural_net.forward_propagation(input_point)
        predicted_prices.append(pred[0])
    
    # Plot the curve
    ax.plot(feature_range, predicted_prices, 'b-', linewidth=2, label='Model Prediction')
    
    # Overlay actual data points for this feature
    ax.scatter(test_inputs[:, feat_idx], test_outputs, alpha=0.3, s=10, c='gray', label='Training Data')
    
    # Convert x-axis to actual values for interpretability
    if feat_idx == 0:  # Square footage
        actual_range = 500 + feature_range * 2500
        ax.set_xlabel('Square Footage', fontsize=12)
        ax2_x = ax.twiny()
        ax2_x.set_xlim(ax.get_xlim())
        ax2_x.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2_x.set_xticklabels(['500', '1125', '1750', '2375', '3000'])
    elif feat_idx == 1:  # Bedrooms
        ax.set_xlabel('Bedrooms (normalized)', fontsize=12)
        ax2_x = ax.twiny()
        ax2_x.set_xlim(ax.get_xlim())
        ax2_x.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2_x.set_xticklabels(['1', '2', '3', '4', '5'])
    else:  # Age
        ax.set_xlabel('Age (normalized)', fontsize=12)
        ax2_x = ax.twiny()
        ax2_x.set_xlim(ax.get_xlim())
        ax2_x.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2_x.set_xticklabels(['0yr', '12yr', '25yr', '37yr', '50yr'])
    
    ax.set_ylabel('Price ($1000s)', fontsize=12)
    ax.set_title(f'Effect of {feature_names[feat_idx]}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('House Price Regression Analysis', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()

# Save
output_path = os.path.join(os.path.dirname(__file__), 'house_price_visualization.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"\nVisualization saved to: {output_path}")