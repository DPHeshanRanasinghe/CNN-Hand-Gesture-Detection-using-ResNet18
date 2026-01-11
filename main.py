import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image

#Configuration & Setup
MODEL_PATH = 'resnet18_rps_best_model.pth'
class_names = ['paper', 'rock', 'scissors']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Loader Function
def load_model(path):
    print(f"Loading model from {path}...")
    
    # Initialize standard ResNet18 architecture
    model = models.resnet18(weights=None)
    
    # Modify the final fully connected layer to match the 3 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    
    # Load the trained weights (State Dictionary)
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        print("Attempting to load as full model...")
        model = torch.load(path, map_location=DEVICE)

    model = model.to(DEVICE)
    model.eval() # Set to evaluation mode (freezes Dropout/BatchNorm)
    return model

# Preprocessing (Standard ResNet Transformations)
# ResNet expects 224x224 images and specific normalization
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Main Webcam Loop
def main():
    model = load_model(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Webcam... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Define a Region of Interest (ROI) box for the hand
        # This helps the model focus by removing messy backgrounds
        # Coordinates: [y1:y2, x1:x2]
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
        roi = frame[100:400, 100:400]

        # Convert BGR (OpenCV) to RGB (PIL/PyTorch)
        img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Preprocess the image
        input_tensor = preprocess(pil_img)
        input_batch = input_tensor.unsqueeze(0) # Add batch dimension (1, 3, 224, 224)
        input_batch = input_batch.to(DEVICE)

        # Inference
        with torch.no_grad():
            output = model(input_batch)
            # Get probabilities using Softmax
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)

        # Get Label
        predicted_label = class_names[predicted_idx.item()]
        conf_score = confidence.item()

        # Display Result on Screen
        text = f"{predicted_label} ({conf_score*100:.1f}%)"
        
        # Color logic: Green if confident, Red if unsure
        color = (0, 255, 0) if conf_score > 0.7 else (0, 0, 255)
        
        cv2.putText(frame, text, (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, color, 2, cv2.LINE_AA)

        cv2.imshow('Rock Paper Scissors Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()