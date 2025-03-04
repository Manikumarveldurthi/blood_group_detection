import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

class BloodGroupCNN(nn.Module):
    def __init__(self, num_classes):
        super(BloodGroupCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def predict_single_image(model, image_path):
    class_labels = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0)
    
    # Move tensor to the same device as model
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        # Get the actual blood type label using the index
        predicted_idx = predicted.item()
        predicted_label = class_labels[predicted_idx]
        confidence_pct = confidence.item() * 100
    
    return {
        "blood_type": predicted_label,
        "confidence": f"{confidence_pct:.1f}%"
    }, image_tensor, predicted_idx

def get_grad_cam(model, image_tensor, class_idx):
    gradients = None
    activations = None
    
    def hook_grad(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]
    
    def hook_activation(module, inp, out):
        nonlocal activations
        activations = out
    
    target_layer = model.conv_layers[-6]
    target_layer.register_forward_hook(hook_activation)
    target_layer.register_backward_hook(hook_grad)
    
    # Enable gradients for this computation
    image_tensor.requires_grad = True
    
    # Forward pass
    output = model(image_tensor)
    
    if output.shape[1] > class_idx:
        class_score = output[0, class_idx]
        
        # Zero gradients
        model.zero_grad()
        
        # Backward pass
        class_score.backward()
        
        # Process gradients and activations
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations = activations.squeeze().detach()
        
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-10  # Add small epsilon to avoid division by zero
        
        return heatmap
    else:
        return np.zeros((64, 64))  

def advanced_xai_plots(image_path, heatmap):
    plots = {}
    
    # Original and Heatmap overlay
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
    heatmap_resized = cv2.resize(heatmap, (64, 64))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
    
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax.set_title("Activation Heatmap")
    ax.axis('off')
    plots['heatmap'] = fig
    
    # 3D Surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]))
    ax.plot_surface(X, Y, heatmap, cmap="viridis")
    ax.set_title("3D Activation Map")
    plots['surface'] = fig
    
    return plots 