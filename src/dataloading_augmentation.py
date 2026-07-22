import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

train_transform = v2.Compose([
    v2.Resize((256, 256)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(degrees=(-45, 45)),
    v2.RandomAffine(degrees=0, scale=(0.8, 1.2)),
    v2.ToImage(),                          # Converts PIL Image to Tensor
    v2.ToDtype(torch.float32, scale=True)  # Normalizes pixel values to [0, 1]
])

from pathlib import Path #object oriented approach to handle filesystem paths

repo_root = Path(__file__).resolve().parent.parent
dataset_path = repo_root / 'data' / 'microplastic-dataset-for-computer-vision' / 'images'

if not dataset_path.exists():
    dataset_path = Path.cwd() / 'data' / 'microplastic-dataset-for-computer-vision' / 'images'


train_dataset = ImageFolder(root=dataset_path, transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

annotations_df = pd.read_csv(dataset_path.parent / 'labels' / '_annotations.csv')

numeric_annotations = annotations_df.select_dtypes(include=[np.number])
annotation_tensors = torch.tensor(numeric_annotations.values, dtype=torch.float32)

