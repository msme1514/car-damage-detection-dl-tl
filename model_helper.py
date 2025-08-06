import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import cv2
from typing import Tuple

# Class labels
class_names = ['Front Breakage', 'Front Crushed', 'Front Normal',
               'Rear Breakage', 'Rear Crushed', 'Rear Normal']

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Custom ResNet classifier with dropout
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.5):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Load model with state_dict
def load_model(model_path="saved_model.pth"):
    model = CarClassifierResNet(num_classes=len(class_names), dropout_rate=0.2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Grad-CAM generator
def generate_gradcam(image_tensor, model, target_layer='layer4'):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Register hooks on the target layer
    for name, module in model.model.named_modules():
        if name == target_layer:
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)

    output = model(image_tensor)
    class_idx = torch.argmax(output).item()

    model.zero_grad()
    output[0, class_idx].backward()

    # Grad-CAM calculation
    grads_val = gradients[0][0]
    acts_val = activations[0][0]
    weights = torch.mean(grads_val, dim=[1, 2])
    cam = torch.zeros(acts_val.shape[1:], dtype=torch.float32)

    for i, w in enumerate(weights):
        cam += w * acts_val[i]
    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max()
    cam = cam.detach().numpy()
    cam = cv2.resize(cam, (224, 224))
    return cam

# Main prediction function
def predict_image(image: Image.Image, model) -> Tuple[str, float, list, np.ndarray]:
    image = image.convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].numpy()
        predicted_index = int(np.argmax(probabilities))
        predicted_class = class_names[predicted_index]
        confidence = float(probabilities[predicted_index])
    
    # Grad-CAM
    cam = generate_gradcam(input_tensor, model)
    return predicted_class, confidence, probabilities, cam
