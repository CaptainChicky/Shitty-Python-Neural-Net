import numpy as np
import plotly.graph_objects as go
import os
import sys
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from neural_network import NeuralNet

# Load the trained model
neural_net = NeuralNet()
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_optimized_red.json')
neural_net.load(model_path)

# Load training data
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'color_data.json')
with open(data_path, 'r') as f:
    data = json.load(f)

train_inputs = np.array(data['RGB_Values'])
train_outputs = np.array(data['Is_Red'])

# Separate red and not-red samples
red_mask = np.array([output[0] == 1 for output in train_outputs])
red_samples = train_inputs[red_mask]
not_red_samples = train_inputs[~red_mask]

print(f"Training data: {len(red_samples)} red, {len(not_red_samples)} not-red")

# Sample test points
num_test_points = 15000
np.random.seed(42)
test_r = np.random.randint(0, 256, num_test_points)
test_g = np.random.randint(0, 256, num_test_points)
test_b = np.random.randint(0, 256, num_test_points)

# Get model predictions
predicted_red = []
predicted_not_red = []

print("Getting model predictions...")
for i in range(num_test_points):
    point = np.array([test_r[i], test_g[i], test_b[i]])
    output = neural_net.forward_propagation(point)
    
    if output[0] > output[1]:
        predicted_red.append([test_r[i], test_g[i], test_b[i]])
    else:
        predicted_not_red.append([test_r[i], test_g[i], test_b[i]])

predicted_red = np.array(predicted_red)
predicted_not_red = np.array(predicted_not_red)

print(f"Model predictions: {len(predicted_red)} red, {len(predicted_not_red)} not-red")

# Create plotly figure
fig = go.Figure()

# Add NOT-RED points first (so red points render on top)
# Sample fewer not-red points since there are way more of them
if len(predicted_not_red) > 0:
    sample_not_red = 2000  # Fewer not-red points
    indices = np.random.choice(len(predicted_not_red), min(sample_not_red, len(predicted_not_red)), replace=False)
    fig.add_trace(go.Scatter3d(
        x=predicted_not_red[indices, 0],
        y=predicted_not_red[indices, 1],
        z=predicted_not_red[indices, 2],
        mode='markers',
        name=f'Model: Not-red ({len(predicted_not_red)} total)',
        marker=dict(
            size=3,  
            color='darkgray', 
            opacity=0.4,  
        )
    ))

# Add RED points
if len(predicted_red) > 0:
    sample_red = 3000  # More red points to see the boundary
    indices = np.random.choice(len(predicted_red), min(sample_red, len(predicted_red)), replace=False)
    fig.add_trace(go.Scatter3d(
        x=predicted_red[indices, 0],
        y=predicted_red[indices, 1],
        z=predicted_red[indices, 2],
        mode='markers',
        name=f'Model: RED ({len(predicted_red)} total)',
        marker=dict(
            size=4,
            color='red',
            opacity=0.7,
            line=dict(color='darkred', width=0.5)
        )
    ))

# Add true sphere as a wireframe
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 20)
x_sphere = 255 + 127 * np.outer(np.cos(u), np.sin(v))
y_sphere = 0 + 127 * np.outer(np.sin(u), np.sin(v))
z_sphere = 0 + 127 * np.outer(np.ones(np.size(u)), np.cos(v))

# Wireframe sphere
for i in range(0, len(u), 1):
    fig.add_trace(go.Scatter3d(
        x=x_sphere[i, :],
        y=y_sphere[i, :],
        z=z_sphere[i, :],
        mode='lines',
        line=dict(color='dodgerblue', width=3),  
        showlegend=(i == 0),
        name='True boundary (r=127)',
        opacity=0.85 
    ))

for j in range(0, len(v), 1):
    fig.add_trace(go.Scatter3d(
        x=x_sphere[:, j],
        y=y_sphere[:, j],
        z=z_sphere[:, j],
        mode='lines',
        line=dict(color='dodgerblue', width=3),  
        showlegend=False,
        opacity=0.85  
    ))

# Add center point
fig.add_trace(go.Scatter3d(
    x=[255],
    y=[0],
    z=[0],
    mode='markers',
    name='Sphere center (255,0,0)',
    marker=dict(
        size=10,
        color='yellow',
        symbol='diamond',
        line=dict(color='black', width=2)
    )
))

# Update layout
fig.update_layout(
    title=dict(
        text='RGB Red Color Classification - 3D Interactive View<br>' +
             '<sub>Red points = Model predicts RED | Gray points = Model predicts NOT-RED | Blue wireframe = True boundary</sub>',
        x=0.5,
        xanchor='center'
    ),
    scene=dict(
        xaxis=dict(title='Red (R)', range=[0, 255], backgroundcolor='rgb(230, 230, 230)'),
        yaxis=dict(title='Green (G)', range=[0, 255], backgroundcolor='rgb(230, 230, 230)'),
        zaxis=dict(title='Blue (B)', range=[0, 255], backgroundcolor='rgb(230, 230, 230)'),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.3)
        ),
        aspectmode='cube'
    ),
    width=1200,
    height=900,
    showlegend=True,
    legend=dict(
        x=0.02,
        y=0.98,
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='black',
        borderwidth=2
    )
)

# Save as HTML
output_path = os.path.join(os.path.dirname(__file__), 'red_color_visualization_3d.html')
fig.write_html(output_path)
print(f"\nInteractive 3D visualization saved to: {output_path}")
print("Open in your web browser - you can rotate, zoom, and pan!")

# Static image
try:
    static_path = os.path.join(os.path.dirname(__file__), 'red_color_visualization_3d.png')
    fig.write_image(static_path, width=1200, height=900)
    print(f"Static image saved to: {static_path}")
except:
    print("(Install kaleido for static image export: pip install kaleido)")

fig.show()

# Calculate accuracy
correct = 0
for i in range(len(train_inputs)):
    output = neural_net.forward_propagation(train_inputs[i])
    predicted_red = output[0] > output[1]
    actual_red = train_outputs[i][0] == 1
    if predicted_red == actual_red:
        correct += 1

accuracy = (correct / len(train_inputs)) * 100
print(f"\nTraining Accuracy: {accuracy:.2f}%")