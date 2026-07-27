
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# basic image transform
img_tfms = v2.Compose([
    v2.Resize((256, 256)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

microscopy_dataset = ImageFolder(
    root=r"data\microplastic-dataset-for-computer-vision\organized_images",
    transform=img_tfms,
)

# optional annotation loading
try:
    annotations_df = pd.read_csv(r"data\microplastic-dataset-for-computer-vision\labels\_annotations.csv")
    numeric_df = annotations_df.apply(pd.to_numeric, errors="coerce")
    annotation_tensors = torch.tensor(numeric_df.to_numpy(dtype=np.float32))
    print("annotations loaded:", annotation_tensors.shape)
except Exception as e:
    print("annotations skipped:", e)
    annotation_tensors = None

# augmentation pipeline
train_transform = v2.Compose([
    v2.Resize((256, 256)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(degrees=(-45, 45)),
    v2.RandomAffine(degrees=0, scale=(0.8, 1.2)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

train_dataset = ImageFolder(
    root=r"data\microplastic-dataset-for-computer-vision\organized_images",
    transform=train_transform,
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print("dataset size:", len(train_dataset))

save_dir = Path(r"data\augmented_images")
save_dir.mkdir(parents=True, exist_ok=True)

for i, (images, labels) in enumerate(train_loader):
    for j, img in enumerate(images):
        save_image(img, save_dir / f"aug_{i}_{j}.png")
