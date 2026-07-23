import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import time

repo_root = Path(__file__).resolve().parent.parent
organized_images_dir = repo_root / "new_data" / "archive" / "train"
manifest_file = repo_root / "new_data" / "archive" / "valid" / "_annotations.csv"
labels_df = pd.read_csv(manifest_file)
print(labels_df.head()) #the labels df now contains the image metadata that the training pipeline uses

image_folder = [organized_images_dir]
images = []
for folder in image_folder: 
    images.extend(os.listdir(folder))

print("Number of images:", len(images)) #the organized image folders are now the source of truth

objects = labels_df.groupby("filename").size() 
print(objects) #counts how many rows each image has in the labels df
print("Max number of labels_df rows per image:", objects.max())

missing = set(labels_df["filename"]) - set(images)
print("Missing images:", len(missing)) #checks whether any labels_df entry points to an image that is no longer present

print("Number images:", len(images)) #the organized dataset count
print("Manifest unique images:", len(labels_df["filename"].unique()))

labels_df["box_width"] = labels_df["xmax"] - labels_df["xmin"]
labels_df["box_height"] = labels_df["ymax"] - labels_df["ymin"]
print("\nBounding Box Width Stats:")
print(labels_df["box_width"].describe())
print("\nBounding Box Height Stats:")
print(labels_df["box_height"].describe())
#plotting boundary box distributions

plt.hist(labels_df["box_width"], bins=20)
plt.title("Microplastic Bounding Box Width Distribution")
plt.xlabel("Width (pixels)")
plt.ylabel("Count")
plt.show()

plt.hist(labels_df["box_height"], bins=20)
plt.title("Microplastic Bounding Box Height Distribution")
plt.xlabel("Height (pixels)")
plt.ylabel("Count")
plt.show()