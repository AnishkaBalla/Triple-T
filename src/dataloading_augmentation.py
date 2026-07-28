import pandas as pd
import torch
from pathlib import Path
from torchvision.transforms import v2
from torchvision.utils import save_image
from PIL import Image

dataset_path = Path("new_data/archive/train")

# augmentation pipeline (5 augments)
train_transform = v2.Compose([
    v2.Resize((256, 256)),
    v2.ColorJitter(
        brightness=0.3,
        contrast=0.2
    ),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])
#we cant do too many transformations bc it doesnt change bounding box coordinates, only image
#load annotations
try:
    annotations_df = pd.read_csv(
        "new_data/archive/train/_annotations.csv"
    )
    print("annotations loaded:", annotations_df.shape)
except Exception as e:
    print("annotations skipped:", e)


#get images
image_files = [
    p for p in dataset_path.rglob("*")
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
]

print("original dataset size:", len(image_files))


#save augmented images in new_data
save_dir = Path("new_data/augmented_train")
save_dir.mkdir(parents=True, exist_ok=True)

num_augments = 2
count = 0

for img_path in image_files:
    image = Image.open(img_path).convert("RGB")

    for i in range(num_augments):
        aug_img = train_transform(image)

        save_image(
            aug_img,
            save_dir / f"{img_path.stem}_aug_{i}.png"
        )
        count += 1

print("augmented images saved:", count) #577*2 = 1154