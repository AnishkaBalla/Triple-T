import pandas as pd #data preprocessing --> the foundation
import numpy as np # mathematical operations
import os # to inspect folders (and interact with operating system)
import cv2 # digital image processing, cv stands for "Computer Vision" 
from PIL import Image #used for handling, editing, and saving images
from pathlib import Path #object oriented approach to handle filesystem paths


repo_root = Path(__file__).resolve().parent.parent
dataset_path = repo_root / 'data' / 'microplastic-dataset-for-computer-vision'

if not dataset_path.exists():
    dataset_path = Path.cwd() / 'data' / 'microplastic-dataset-for-computer-vision'

images_dir = dataset_path / 'images'
labels_dir = dataset_path / 'labels'

df = pd.read_csv(labels_dir / "_annotations.csv")
print(df.head())
print(df.shape)
print(df.isna())
print(df[df.duplicated()])
# all is well! no duplicates or null values within the dataset to clean.
print(df.info())
print(df.describe())
#get a list of all image and annotation files
images = {file.stem for file in images_dir.glob('*.*') if file.suffix.lower() in ('.png', '.jpg', '.jpeg')}
annotations = {file.stem for file in labels_dir.glob('.csv')}
#Pandas series for images and annotations to verify
img_series = pd.Series(list(images), name='image_name')
annot_series = pd.Series(list(annotations), name='annotation_name')
# find missing annotations for images, and vice versa
missing_annotations = img_series[~img_series.isin(annot_series)]
missing_images = annot_series[~annot_series.isin(img_series)]

print(f"Images missing annotation files -> {len(missing_annotations)}")
print(f"Annotations missing image files -> {len(missing_images)}")
# yay! both values are 0, so let's continue, because every image has a corresponding annotation file. we have finally verified this.

#now lets remove any remaining corrupted / duplicate files just to make sure:
valid_images = []
for index, row in df.iterrows():
    filename = str(row['filename']).strip()
    img_path = images_dir / filename
    if not img_path.exists():
        img_path = images_dir / f"{filename}.jpg"
    if not img_path.exists():
        continue

    if img_path.stem in valid_images:
        os.remove(img_path)
        continue

    try:
        with Image.open(img_path) as img:
            img.verify()
        valid_images.append(img_path.stem)
    except Exception as e:
        print(f"Removing corrupted image: {img_path.name} | Error: {e}")
        os.remove(img_path)
        
#update pandas DataFrame to only include valid, non-corrupted images
cleaned_df = df[df['filename'].isin(valid_images)].drop_duplicates()

# save cleaned annotations
cleaned_df.to_csv(labels_dir / 'cleaned_annotations.csv', index=False)
