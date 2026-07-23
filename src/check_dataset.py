import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import time

image_folder = ["data/microplastic-dataset-for-computer-vision/images", "data/microplastic-dataset-for-computer-vision/images/ClassA"]
annotation_file = "data/microplastic-dataset-for-computer-vision/labels/_annotations.csv"
annotations = pd.read_csv(annotation_file)
print(annotations.head()) #since the csv from the dataset is in a different format (VOC), it looks like filename | xmin | ymin | xmax | ymax -> 4 values making up the boundary box

images = []
for folder in image_folder: 
    images.extend(os.listdir(folder))

print("Number of images:", len(images)) #503 images

objects = annotations.groupby("filename").size() 
print(objects) #counts the number of microplastics per image
print("Max number of microplastics in an image:", objects.max()) #the max is 28

missing = set(annotations["filename"]) - set(images)
print("Missing images:", len(missing)) #203 images appear in annotations but not in image folders

print("Number images:", len(images)) #579 images in dataset
print("CSV unique images:", len(annotations["filename"].unique())) #204 unique labeled images (meaning the rest have no microplastic - assumption)

annotations["box_width"] = annotations["xmax"] - annotations["xmin"]
annotations["box_height"] = annotations["ymax"] - annotations["ymin"]
print("\nBounding Box Width Stats:")
print(annotations["box_width"].describe())
print("\nBounding Box Height Stats:")
print(annotations["box_height"].describe())
#plotting boundary box distributions

plt.hist(annotations["box_width"], bins=20)
plt.title("Microplastic Bounding Box Width Distribution")
plt.xlabel("Width (pixels)")
plt.ylabel("Count")
plt.show()

plt.hist(annotations["box_height"], bins=20)
plt.title("Microplastic Bounding Box Height Distribution")
plt.xlabel("Height (pixels)")
plt.ylabel("Count")
plt.show()