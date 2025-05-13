import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import gdown
import os

# Page setup
st.set_page_config(page_title="EuroSAT Classifier", layout="centered")
st.title("🌍 EuroSAT Land Cover Classifier")
st.write("Upload a satellite image to get the land cover classification")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names
class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

# Image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Model definition
class EuroSATModel(nn.Module):
    def __init__(self, num_classes=10):
        super(EuroSATModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# Load model from Google Drive with caching
@st.cache_resource
def load_model():
    try:
        file_id = "1uT9vXYEEFDxKAThf8A371RRSnD9Kq44o"
        url = f"https://drive.google.com/uc?id={file_id}"
        output = "eurosat_resnet18.pth"
        
        if not os.path.exists(output):
            with st.spinner('Downloading model from Google Drive...'):
                gdown.download(url, output, quiet=False)
        
        # Initialize model first
        model = EuroSATModel(num_classes=10).to(device)
        
        # Load only the state_dict
        state_dict = torch.load(output, map_location=device)
        
        # Handle layer names if model was saved on multi-GPU
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):  # If saved on multi-GPU
                k = k[7:]  # Remove 'module.' prefix
            new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        model.eval()
        return model
    
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# Load the model
model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    try:
        # Open and display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Transform image
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            class_name = class_names[predicted.item()]
            confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item()
        
        # Display results
        st.success(f"✅ **Prediction:** {class_name}")
        st.info(f"📊 **Confidence:** {confidence * 100:.2f}%")
        
        # Optional: show all class probabilities
        with st.expander("Show all class probabilities"):
            probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
            for i, (class_name, prob) in enumerate(zip(class_names, probs)):
                st.write(f"{class_name}: {prob * 100:.2f}%")
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")