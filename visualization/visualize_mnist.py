import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from neural_network import NeuralNet

# Load the trained model
neural_net = NeuralNet()
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_mnist.json')
neural_net.load(model_path)

# Load test data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mnist_test.json')
with open(data_path, 'r') as f:
    data = json.load(f)

test_inputs = np.array(data['Input_Values'])
test_outputs = np.array(data['Output_Values'])

print(f"Loaded {len(test_inputs)} test samples")

# Make predictions
predictions = []
actuals = []
confidences = []
output_distributions = []

for i in range(len(test_inputs)):
    prediction = neural_net.forward_propagation(test_inputs[i])
    predicted_digit = np.argmax(prediction)
    actual_digit = np.argmax(test_outputs[i])
    confidence = prediction[predicted_digit]
    
    predictions.append(predicted_digit)
    actuals.append(actual_digit)
    confidences.append(confidence)
    output_distributions.append(prediction)

predictions = np.array(predictions)
actuals = np.array(actuals)
confidences = np.array(confidences)
output_distributions = np.array(output_distributions)
correct = (predictions == actuals)

accuracy = np.mean(correct) * 100
print(f"Accuracy: {accuracy:.2f}%")

incorrect_indices = np.where(~correct)[0]

############################################################################################################
# t-SNE DIMENSIONALITY REDUCTION
############################################################################################################
print("Computing t-SNE projection (this may take a minute)...")

n_samples = min(6000, len(test_inputs))
sample_indices = np.random.choice(len(test_inputs), n_samples, replace=False)

pca = PCA(n_components=50)
inputs_pca = pca.fit_transform(test_inputs[sample_indices])

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
inputs_2d = tsne.fit_transform(inputs_pca)

print("t-SNE complete!")

############################################################################################################
# VISUAL MNIST ANALYSIS
############################################################################################################

fig = plt.figure(figsize=(22, 14))
gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.4)

# ============================================
# PLOT 1: t-SNE 2D PROJECTION (LEFT SIDE - LARGE)
# ============================================
ax1 = fig.add_subplot(gs[0:2, 0:2])

# Better distinct colors
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', 
          '#ffff33', '#a65628', '#f781bf', '#999999', '#00CED1']

# Plot each digit
for digit in range(10):
    digit_mask = actuals[sample_indices] == digit
    ax1.scatter(inputs_2d[digit_mask, 0], inputs_2d[digit_mask, 1],
               c=colors[digit], label=f'{digit}', s=25, alpha=0.7, 
               edgecolors='black', linewidths=0.3)

# Highlight mistakes with RED CIRCLES - avoid overlaps
mistake_mask = np.isin(sample_indices, incorrect_indices)
mistake_sample_indices = sample_indices[mistake_mask]

plotted_positions = []
min_distance = 5.5  # Minimum distance to avoid overlap

for idx in mistake_sample_indices:
    pos = np.where(sample_indices == idx)[0][0]
    actual = actuals[idx]
    pred = predictions[idx]
    
    position = inputs_2d[pos]
    
    # Check if too close to already plotted mistakes
    too_close = False
    for prev_pos in plotted_positions:
        distance = np.linalg.norm(position - prev_pos)
        if distance < min_distance:
            too_close = True
            break
    
    if too_close:
        continue
    
    # Red circle (smaller)
    ax1.scatter(position[0], position[1],
               marker='o', s=180, facecolors='red', edgecolors='darkred', 
               linewidths=2.5, zorder=10, alpha=0.85)
    
    # Predicted digit inside (smaller font)
    ax1.text(position[0], position[1], str(pred),
            fontsize=10, fontweight='bold', color='white', ha='center', va='center',
            zorder=11)
    
    plotted_positions.append(position)

ax1.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
ax1.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
ax1.set_title('2D Projection of 784D MNIST Space\n(Colored dots = digits, Red circles = mistakes with predicted # inside)', 
              fontsize=13, fontweight='bold')

legend = ax1.legend(loc='upper right', fontsize=10, ncol=2, framealpha=0.9, 
                   edgecolor='black', title='Digits', title_fontsize=11)
for handle in legend.legend_handles:
    handle.set_sizes([100])

ax1.grid(True, alpha=0.2)

# ============================================
# PLOT 2: CONFIDENCE DISTRIBUTION (LOG SCALE) - TOP RIGHT
# ============================================
ax2 = fig.add_subplot(gs[0, 2:4])

correct_conf = confidences[correct]
incorrect_conf = confidences[~correct]

bins = np.linspace(0, 1.0, 25)
counts_correct, bin_edges = np.histogram(correct_conf, bins=bins)
counts_incorrect, _ = np.histogram(incorrect_conf, bins=bins)

bar_width = (bin_edges[1] - bin_edges[0]) / 2.5
x_correct = bin_edges[:-1]
x_incorrect = bin_edges[:-1] + bar_width

ax2.bar(x_correct, counts_correct, width=bar_width, alpha=0.8, color='green',
       label=f'Correct ({len(correct_conf)})', edgecolor='black', linewidth=0.5)
ax2.bar(x_incorrect, counts_incorrect, width=bar_width, alpha=0.8, color='red',
       label=f'Incorrect ({len(incorrect_conf)})', edgecolor='black', linewidth=0.5)

ax2.set_yscale('log')
ax2.set_xlabel('Confidence', fontsize=11, fontweight='bold')
ax2.set_ylabel('Count (Log Scale)', fontsize=11, fontweight='bold')
ax2.set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(True, alpha=0.3, which='both', axis='y', linestyle='--')
ax2.axvspan(0, 0.5, alpha=0.15, color='orange')

# ============================================
# PLOT 3: SOFTMAX HEATMAP - SECOND ROW RIGHT
# ============================================
ax3 = fig.add_subplot(gs[1, 2:4])

sample_incorrect = incorrect_indices[:40] if len(incorrect_indices) >= 40 else incorrect_indices
if len(sample_incorrect) > 0:
    mistake_outputs = output_distributions[sample_incorrect]
    
    im = ax3.imshow(mistake_outputs.T, cmap='RdYlGn_r', aspect='auto', 
                    interpolation='nearest', vmin=0, vmax=1)
    
    ax3.set_xlabel('Mistake Sample Index', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Output Neuron (Digit)', fontsize=10, fontweight='bold')
    ax3.set_title('Network Outputs for First 40 Misclassified Samples\n(Blue Box=Actual | Red Box=Predicted)', 
                  fontsize=11, fontweight='bold')
    ax3.set_yticks(range(10))
    ax3.set_yticklabels([str(i) for i in range(10)])
    
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Softmax Output', rotation=270, labelpad=15, fontweight='bold', fontsize=9)
    
    # Mark actual (blue) and predicted (red)
    for i, idx in enumerate(sample_incorrect):
        actual = actuals[idx]
        pred = predictions[idx]
        ax3.add_patch(plt.Rectangle((i-0.45, actual-0.45), 0.9, 0.9, 
                                    fill=False, edgecolor='blue', linewidth=2.5))
        ax3.add_patch(plt.Rectangle((i-0.45, pred-0.45), 0.9, 0.9, 
                                    fill=False, edgecolor='red', linewidth=2.5))
else:
    ax3.text(0.5, 0.5, 'PERFECT!', ha='center', va='center',
            fontsize=16, fontweight='bold', color='green', transform=ax3.transAxes)
    ax3.axis('off')

# ============================================
# PLOT 4: WHERE DID THE CORRECT ANSWER RANK?
# ============================================
ax4 = fig.add_subplot(gs[2, 2])

if len(incorrect_indices) > 0:
    # For each mistake, find where the correct digit ranked
    ranks = []
    for idx in incorrect_indices:
        outputs = output_distributions[idx]
        actual = actuals[idx]
        
        # Sort outputs in descending order
        sorted_indices = np.argsort(outputs)[::-1]
        
        # Find rank of actual digit (1-indexed)
        rank = np.where(sorted_indices == actual)[0][0] + 1
        ranks.append(rank)
    
    # Count how many at each rank
    rank_counts = {}
    for rank in range(2, 11):
        rank_counts[rank] = np.sum(np.array(ranks) == rank)
    
    # Plot
    rank_positions = list(rank_counts.keys())
    rank_values = list(rank_counts.values())
    
    bars = ax4.bar(rank_positions, rank_values, color='lightcoral', 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Highlight rank 2 (second choice) in orange
    if len(bars) > 0:
        bars[0].set_color('steelblue')
        bars[0].set_edgecolor('black')
    
    # Add counts on bars
    for pos, val in zip(rank_positions, rank_values):
        ax4.text(pos, val + 0.5, str(val), ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')

    # Set y-axis limit to prevent label cutoff
    max_val = max(rank_values) if rank_values else 10
    ax4.set_ylim(0, max_val * 1.1)  # 10% extra space above highest bar
    
    ax4.set_xlabel('Rank of CORRECT Digit', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Number of Mistakes', fontsize=10, fontweight='bold')
    
    # Clearer title explaining what it means
    second_choice_count = rank_counts.get(2, 0)
    pct = (second_choice_count / len(incorrect_indices)) * 100
    ax4.set_title(f'How Close Was Network to Being Right?\n({second_choice_count} mistakes had 2nd choice correct ~{pct:.1f}%)', 
                  fontsize=10, fontweight='bold')
    ax4.set_xticks(rank_positions)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add explanation annotation
    ax4.text(0.976, 0.97, 'Rank 2 = "Almost right!"\nRank 10 = "Way off!"', 
            transform=ax4.transAxes, fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
else:
    ax4.text(0.5, 0.5, 'No mistakes!', ha='center', va='center',
            fontsize=14, fontweight='bold', color='green', transform=ax4.transAxes)
    ax4.axis('off')

# ============================================
# PLOT 5: PER-DIGIT ACCURACY (90-100%)
# ============================================
ax5 = fig.add_subplot(gs[2, 3])

per_digit_accuracy = []
for digit in range(10):
    digit_indices = np.where(actuals == digit)[0]
    if len(digit_indices) > 0:
        digit_accuracy = np.mean(correct[digit_indices]) * 100
    else:
        digit_accuracy = 0
    per_digit_accuracy.append(digit_accuracy)

colors_bar = ['green' if acc >= 95 else 'orange' if acc >= 92 else 'red' 
              for acc in per_digit_accuracy]

bars = ax5.bar(range(10), per_digit_accuracy, color=colors_bar, 
               edgecolor='black', linewidth=1.2, alpha=0.8)

for i, acc in enumerate(per_digit_accuracy):
    ax5.text(i, acc + 0.08, f'{acc:.1f}', ha='center', va='bottom', 
            fontsize=7.5, fontweight='bold')

ax5.set_xlabel('Digit', fontsize=10, fontweight='bold')
ax5.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
ax5.set_title('Per-Digit Accuracy\n(Zoomed: 90-100%)', fontsize=11, fontweight='bold')
ax5.set_xticks(range(10))
ax5.set_ylim(90, 100)
ax5.grid(axis='y', alpha=0.3, linestyle='--')
ax5.axhline(y=accuracy, color='blue', linestyle='--', linewidth=1, alpha=0.6, label='Overall Accuracy')
ax5.legend(fontsize=7.5, loc='upper right')

# ============================================
# PLOT 6: 15 WORST MISTAKES (3x5 GRID - MOVED UP)
# ============================================
ax6 = fig.add_subplot(gs[2, 0:2])
ax6.axis('off')
ax6.set_title('15 Most Confident Wrong Predictions \nActual → Predicted (Confidence)', 
              fontsize=12, fontweight='bold', color='darkred')

if len(incorrect_indices) > 0:
    worst_mistakes = incorrect_indices[np.argsort(confidences[incorrect_indices])[-15:]]

    x_offset = 0.125
    y_start = 0.235
    
    for idx, sample_idx in enumerate(worst_mistakes):
        row = idx // 5
        col = idx % 5
        
        # Moved up by increasing y position
        ax_img = fig.add_axes([x_offset + col * 0.065, y_start - row * 0.085, 0.08, 0.055])
        
        img = test_inputs[sample_idx].reshape(28, 28)
        ax_img.imshow(img, cmap='gray')
        ax_img.axis('off')
        
        pred = predictions[sample_idx]
        actual = actuals[sample_idx]
        conf = confidences[sample_idx]
        
        ax_img.set_title(f'{actual}→{pred} ({conf:.5f})', 
                        fontsize=8, fontweight='bold', color='darkred')
        
        for spine in ax_img.spines.values():
            spine.set_edgecolor('darkred')
            spine.set_linewidth(2.5)
else:
    ax6.text(0.5, 0.5, 'PERFECT!', ha='center', va='center',
            fontsize=20, fontweight='bold', color='green', transform=ax6.transAxes)

plt.suptitle(f'MNIST Visualization - Accuracy: {accuracy:.2f}% ({np.sum(correct)}/{len(correct)} correct)',
             fontsize=17, fontweight='bold', y=0.94)

# Save
output_path = os.path.join(os.path.dirname(__file__), 'mnist_visualization.png')
plt.savefig(output_path, dpi=200, bbox_inches='tight')
plt.show()

print(f"\nVisualization saved to: {output_path}")