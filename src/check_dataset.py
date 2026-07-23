import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import time

repo_root = Path(__file__).resolve().parent.parent
organized_images_dir = repo_root / "data" / "microplastic-dataset-for-computer-vision" / "organized_images"
manifest_file = organized_images_dir / "dataset_manifest.csv"
manifest = pd.read_csv(manifest_file)
print(manifest.head()) #the manifest now contains the image metadata that the training pipeline uses

image_folder = [organized_images_dir, organized_images_dir / "ClassA"]
images = []
for folder in image_folder: 
    images.extend(os.listdir(folder))

print("Number of images:", len(images)) #the organized image folders are now the source of truth

objects = manifest.groupby("filename").size() 
print(objects) #counts how many rows each image has in the manifest
print("Max number of manifest rows per image:", objects.max())

missing = set(manifest["filename"]) - set(images)
print("Missing images:", len(missing)) #checks whether any manifest entry points to an image that is no longer present

print("Number images:", len(images)) #the organized dataset count
print("Manifest unique images:", len(manifest["filename"].unique()))

manifest["box_width"] = manifest["xmax"] - manifest["xmin"]
manifest["box_height"] = manifest["ymax"] - manifest["ymin"]
print("\nBounding Box Width Stats:")
print(manifest["box_width"].describe())
print("\nBounding Box Height Stats:")
print(manifest["box_height"].describe())
#plotting boundary box distributions

plt.hist(manifest["box_width"], bins=20)
plt.title("Microplastic Bounding Box Width Distribution")
plt.xlabel("Width (pixels)")
plt.ylabel("Count")
plt.show()

plt.hist(manifest["box_height"], bins=20)
plt.title("Microplastic Bounding Box Height Distribution")
plt.xlabel("Height (pixels)")
plt.ylabel("Count")
plt.show()