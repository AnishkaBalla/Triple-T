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

rows_before = len(df)
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

# clean the annotations table without dropping rows just because the image file names
# do not match the current folder layout exactly
cleaned_df = df.drop_duplicates().copy()
cleaned_df = cleaned_df.dropna(subset=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
cleaned_df = cleaned_df[cleaned_df['filename'].astype(str).str.strip() != '']

rows_after = len(cleaned_df)
duplicates_removed = rows_before - rows_after

print(f"Rows before cleaning: {rows_before}")
print(f"Rows after cleaning: {rows_after}")
print(f"Duplicate rows removed: {duplicates_removed}")

# save cleaned annotations
cleaned_df.to_csv(labels_dir / 'cleaned_annotations.csv', index=False)

new_df = pd.read_csv(labels_dir / 'cleaned_annotations.csv')
print(new_df.shape)
print(new_df.head())