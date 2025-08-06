import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

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

# Define the custom model class (same as training)
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.5):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace FC layer
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Load model with architecture
def load_model(model_path="saved_model.pth"):
    model = CarClassifierResNet(num_classes=len(class_names), dropout_rate=0.2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Predict single image
def predict_image(image: Image.Image, model):
    image = image.convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        predicted_class = class_names[predicted_index]
        confidence = probabilities[0][predicted_index].item()
    return predicted_class, confidence
