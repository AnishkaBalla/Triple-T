import os
import random
import shutil
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def convert_csv_to_yolo(csv_path, images_dir, labels_out_dir):
    """
    Reads a Pascal-VOC-style annotations CSV (filename, class, xmin, ymin, xmax, ymax)
    and writes one YOLO-format .txt label file per image into labels_out_dir.
    Returns the sorted list of class names (their index = YOLO class_id).
    """
    df = pd.read_csv(csv_path)
    class_names = sorted(df["class"].unique())
    class_to_id = {name: i for i, name in enumerate(class_names)}

    os.makedirs(labels_out_dir, exist_ok=True)

    for filename, group in df.groupby("filename"):
        filename = filename.strip()
        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            print(f"WARNING: image not found, skipping: {img_path}")
            continue

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        lines = []
        for _, row in group.iterrows():
            class_id = class_to_id[row["class"]]
            x_center = ((row["xmin"] + row["xmax"]) / 2) / img_w
            y_center = ((row["ymin"] + row["ymax"]) / 2) / img_h
            box_w = (row["xmax"] - row["xmin"]) / img_w
            box_h = (row["ymax"] - row["ymin"]) / img_h
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        label_filename = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(labels_out_dir, label_filename), "w") as f:
            f.write("\n".join(lines))

    print(f"Converted {df['filename'].nunique()} images. Classes: {class_names}")
    return class_names


def draw_boxes(image_path, label_path):
    """
    Displays an image with YOLO-format bounding boxes drawn on top.
    label_path should point to a .txt file with lines: class_id x_center y_center width height
    """
    img = Image.open(image_path)
    img_w, img_h = img.size

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                class_id, x_center, y_center, w, h = map(float, line.split())
                box_w = w * img_w
                box_h = h * img_h
                x_min = (x_center * img_w) - (box_w / 2)
                y_min = (y_center * img_h) - (box_h / 2)
                rect = patches.Rectangle(
                    (x_min, y_min), box_w, box_h,
                    linewidth=1.5, edgecolor="blue", facecolor="none"
                )
                ax.add_patch(rect)
    else:
        print(f"No label file found for {image_path}")

    plt.axis("off")
    plt.show()


def split_dataset(images_dir, labels_dir, output_dir, split=(0.7, 0.2, 0.1), seed=42):
    """
    Splits images (and their matching labels) into train/valid/test folders
    under output_dir, each with images/ and labels/ subfolders.
    """
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")]
    random.seed(seed)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * split[0])
    n_val = int(n * split[1])

    splits = {
        "train": images[:n_train],
        "valid": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }

    for split_name, files in splits.items():
        img_out = os.path.join(output_dir, split_name, "images")
        lbl_out = os.path.join(output_dir, split_name, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for filename in files:
            shutil.copy(os.path.join(images_dir, filename), os.path.join(img_out, filename))
            label_name = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_name)
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(lbl_out, label_name))

    print(f"Split {n} images -> train:{len(splits['train'])} valid:{len(splits['valid'])} test:{len(splits['test'])}")

# 
def draw_bounding_boxes(images_dir, labels_dir, num_samples = 15):
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")]
    random.seed(42)
    sampled_images = random.sample(images, min(num_samples, len(images)))

    for filename in sampled_images:
        image_path = os.path.join(images_dir, filename)
        label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")

        draw_boxes(image_path, label_path)


if __name__ == "__main__":
    print("Testing draw_bounding_boxes() with sample images from train dataset:")
    draw_bounding_boxes(
        images_dir="data/microplastic-dataset-for-computer-vision/train/images",
        labels_dir="data/microplastic-dataset-for-computer-vision/train/labels"
    )


# 


