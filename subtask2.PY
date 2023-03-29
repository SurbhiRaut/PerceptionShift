import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision.transforms.functional import resize
from torchvision import transforms
from tqdm import tqdm
from deeplabv3plus import DeepLabv3Plus


# Load DeepLabv3+ model
model = DeepLabv3Plus(backbone="resnet50", num_classes=1)

# Load model weights
weights_path = "path/to/model/weights.pt"
model.load_state_dict(torch.load(weights_path))

# Define image size
img_size = 640

# Load images from dataset
dataset_dir = Path("path/to/dataset")
image_paths = list(dataset_dir.glob("*.jpg"))

# Initialize empty list to store predicted auras
predicted_auras = []

# Loop through all images in dataset
for image_path in tqdm(image_paths):
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Resize image
    image = resize(image, img_size)
    
    # Convert image to tensor
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    
    # Predict segmentation mask
    with torch.no_grad():
        mask = model(image_tensor)[0][0]
    
    # Convert mask to numpy array
    mask = mask.detach().cpu().numpy()
    
    # Threshold mask to binarize it
    mask = (mask > 0.5).astype(np.uint8)
    
    # Find contours in mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate number of pixels occupied by each human
    for contour in contours:
        # Convert contour to binary mask
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [contour], 0, 1, -1)
        
        # Calculate area of binary mask
        area = np.sum(mask)
        
        # Calculate aura of human
        aura = round((area ** (3/2)) / (36 * np.pi), 2)
        
        # Add aura to list of predicted auras
        predicted_auras.append(aura)

# Save predicted auras in CSV file
df = pd.DataFrame(predicted_auras, columns=["Human_Auras"])
df.to_csv("Human_Auras.csv", index=False)
