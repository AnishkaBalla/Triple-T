import numpy as np  # math for annotation table
import pandas as pd  # Import pandas so the annotation CSV can be read as a table.
import torch  # data converted to tensors for training
from torchvision.transforms import v2  # import torchvision transforms so images can be resized and normalized.
from torchvision.datasets import ImageFolder  #import ImageFolder so the organized class folders can be loaded as a dataset.
from torch.utils.data import DataLoader  # import DataLoader so batches of images can be fed to the model.

train_transform = v2.Compose([  # create a single transform pipeline for all training images.
    v2.Resize((256, 256)),  # resize every image to a fixed shape so the model sees consistent input sizes.
    v2.ToImage(),  #convert the image into a tensor-friendly image object.
    v2.ToDtype(torch.float32, scale=True)  # convert the image values to float32 and scale them into the range 0 to 1.
])

from pathlib import Path  # import Path so filesystem locations can be handled in a portable way.

repo_root = Path(__file__).resolve().parent.parent  # resolve the repository root from the current script location.
dataset_root = repo_root / 'data' / 'microplastic-dataset-for-computer-vision'  # build the dataset folder path relative to the repository root.
organized_dataset_path = dataset_root / 'organized_images'  # point -> organized class-based image directory if it exists.
dataset_path = organized_dataset_path if organized_dataset_path.exists() else dataset_root / 'images'  #use the organized folders when present, otherwise fall back to the original image folder.

if not dataset_path.exists():  #if neither the organized folder nor the original folder exists at the repository path, fall back to the current working directory.
    dataset_path = Path.cwd() / 'data' / 'microplastic-dataset-for-computer-vision' / 'images'  # build relative fallback path from the current working directory.


train_dataset = ImageFolder(root=dataset_path, transform=train_transform)  # PyTorch dataset from the selected image root folder.

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  #data loader that batches images and shuffles them during training.

annotations_df = pd.read_csv(dataset_path.parent / 'labels' / '_annotations.csv')  #read the annotation CSV from the dataset labels folder.

numeric_annotations = annotations_df.select_dtypes(include=[np.number])  #keep only the numeric columns from the annotation table for tensor conversion.
annotation_tensors = torch.tensor(numeric_annotations.values, dtype=torch.float32)  # convert the numeric annotation values into a PyTorch tensor.

