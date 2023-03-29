import cv2
import pandas as pd

# Load the CSV file with pixel information for each human
human_pixels = pd.read_csv("Human_Pixels.csv")

# Load the pre-trained human segmentation model
model = cv2.dnn.readNetFromTensorflow("human_segmentation.pb")

# Create empty lists to store the volume and aura of each human
volumes = []
auras = []

# Loop over each human in each image
for i, row in human_pixels.iterrows():
    # Load the image
    image = cv2.imread(row["Image_Path"])

    # Extract the human pixels from the image
    x, y, w, h, depth = row["X"], row["Y"], row["Width"], row["Height"], row["Depth"]
    human_pixels = image[y:y+h, x:x+w]

    # Use the segmentation model to segment the human pixels
    blob = cv2.dnn.blobFromImage(human_pixels, 1/255.0, (512, 512), swapRB=True, crop=False)
    model.setInput(blob)
    mask = model.forward()[0, 0]

    # Calculate the number of pixels occupied by the human
    num_pixels = (mask > 0.5).sum()

    # Calculate the pixel area and convert depth to meters
    pixel_area = (w * h) / num_pixels
    depth_m = depth / 1000.0

    # Calculate the volume of the human
    volume = (num_pixels * pixel_area) * depth_m
    volumes.append(volume)

    # Calculate the aura of the human
    surface_area = (w * h) + (w * depth_m) + (h * depth_m)
    aura = volume / surface_area
    auras.append(aura)

# Create a DataFrame to store the results
results = pd.DataFrame({
    "Image_ID": human_pixels["Image_ID"],
    "Human_ID": human_pixels["Human_ID"],
    "Volume": volumes,
    "Aura": auras
})

# Save the results to CSV files
results.to_csv("Human_Volumes.csv", index=False)
results[["Image_ID", "Human_ID", "Aura"]].to_csv("Human_Auras.csv", index=False)
