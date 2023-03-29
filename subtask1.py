import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision.transforms.functional import resize
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.torch_utils import time_synchronized


# Load YOLOv5 model
model = attempt_load("yolov5s.pt", map_location=torch.device("cpu"))

# Define image size
img_size = 640

# Define confidence threshold
conf_thres = 0.5

# Define iou threshold
iou_thres = 0.5

# Load images from dataset
dataset_dir = Path("path/to/dataset")
image_paths = list(dataset_dir.glob("*.jpg"))

# Define class labels
class_labels = ["person", "car", "motorbike", "bus", "truck", "traffic light", "stop sign"]

# Initialize empty list to store predicted labels
predicted_labels = []

# Loop through all images in dataset
for image_path in image_paths:
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Resize image
    image = resize(image, img_size)
    
    # Convert image to tensor
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    
    # Predict bounding boxes and class labels
    with torch.no_grad():
        t1 = time_synchronized()
        results = model(image_tensor, conf_thres=conf_thres, iou_thres=iou_thres)
        t2 = time_synchronized()
        print(f"Inference time: {(t2 - t1):.2f}s")
    
    # Extract class labels and store in list
    labels = []
    for result in results:
        labels.append(class_labels[int(result[-1])])
    
    # Add labels to list of predicted labels
    predicted_labels.append(labels)

# Save predicted labels in CSV file
df = pd.DataFrame(predicted_labels, columns=["Object_Labels"])
df.to_csv("Object_Labels.csv", index=False)
