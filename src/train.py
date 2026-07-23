import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# this part helps the script run well
# it also installs sklearn if it is missing
try:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split

from cnn_model import CustomCNN
from dataloading_augmentation import dataset_path, train_transform

# this file trains a simple cnn for object box prediction
# it also shows easy-to-read plots so the student can understand what is happening
ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = dataset_path
LABELS_PATH = dataset_path.parent / "labels" / "cleaned_annotations.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 0.001


def make_loader(paths, labels_df, batch_size, shuffle):
    # this builds the pytorch dataset from the image files and labels
    ds = BoxDataset(paths, labels_df, transform=train_transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


class BoxDataset(Dataset):
    # this class gives one image and one target box at a time
    def __init__(self, image_paths, labels_df, transform=None):
        self.image_paths = image_paths
        self.labels_df = labels_df
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # this opens the image and makes it ready for the cnn
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # this looks for the box labels for this image
        row = self.labels_df[self.labels_df["filename"] == path.name]
        if row.empty:
            # if there is no box, the image is treated as having no object
            target = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        else:
            # this turns the box coordinates into normalized values the model can learn
            r = row.iloc[0]
            x1, y1, x2, y2 = r["xmin"], r["ymin"], r["xmax"], r["ymax"]
            width, height = r["width"], r["height"]
            cx = ((x1 + x2) / 2) / width
            cy = ((y1 + y2) / 2) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            target = torch.tensor([cx, cy, bw, bh, 1.0], dtype=torch.float32)

        return image, target


def plot_training_history(train_losses, val_losses):
    # this plot shows whether the model is learning or getting stuck
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train loss", marker="o")
    plt.plot(val_losses, label="validation loss", marker="o")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("training and validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_history.png")
    print("saved training curve to training_history.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred):
    # this heatmap shows how often the model was right or wrong
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], ["no object", "object"])
    plt.yticks([0, 1], ["no object", "object"])
    plt.xlabel("predicted label")
    plt.ylabel("true label")
    plt.title("confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("saved confusion matrix to confusion_matrix.png")
    plt.close()


def evaluate_model(model, loader, device):
    # this checks the model after training and prints clear evaluation numbers
    model.eval()
    all_true = []
    all_pred = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            preds = model(images)[:, 0, :]
            probs = torch.sigmoid(preds[:, 4]).cpu()
            pred_labels = (probs >= 0.5).long()
            true_labels = (targets[:, 4] >= 0.5).long().cpu()
            all_true.append(true_labels)
            all_pred.append(pred_labels)

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("accuracy:", round(acc, 4))
    print("precision:", round(prec, 4))
    print("recall:", round(rec, 4))
    print("f1 score:", round(f1, 4))
    plot_confusion_matrix(y_true, y_pred)


def main():
    # this is the full training flow from start to finish
    torch.manual_seed(42)
    labels_df = pd.read_csv(LABELS_PATH)
    image_paths = [p for p in IMAGES_DIR.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    train_files, test_files = train_test_split(image_paths, test_size=0.2, random_state=42)
    train_loader = make_loader(train_files, labels_df, BATCH_SIZE, shuffle=True)
    test_loader = make_loader(test_files, labels_df, BATCH_SIZE, shuffle=False)

    model = CustomCNN(max_objects=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    box_loss_fn = torch.nn.SmoothL1Loss()
    conf_loss_fn = torch.nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        # this is the training part of the loop
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            preds = model(images)[:, 0, :]
            pred_boxes = preds[:, :4]
            pred_conf_logits = preds[:, 4]
            target_boxes = targets[:, :4]
            target_conf = targets[:, 4]

            # box loss is only used when the image really has an object
            box_loss_raw = F.smooth_l1_loss(pred_boxes, target_boxes, reduction="none").mean(dim=1)
            box_loss = (box_loss_raw * target_conf).mean()
            conf_loss = conf_loss_fn(pred_conf_logits, target_conf)
            loss = conf_loss + box_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # this is the validation part of the loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                preds = model(images)[:, 0, :]
                pred_boxes = preds[:, :4]
                pred_conf_logits = preds[:, 4]
                target_boxes = targets[:, :4]
                target_conf = targets[:, 4]

                box_loss_raw = F.smooth_l1_loss(pred_boxes, target_boxes, reduction="none").mean(dim=1)
                box_loss = (box_loss_raw * target_conf).mean()
                conf_loss = conf_loss_fn(pred_conf_logits, target_conf)
                loss = conf_loss + box_loss
                val_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(test_loader))
        print(f"epoch {epoch + 1}/{EPOCHS} | train loss: {train_losses[-1]:.4f} | validation loss: {val_losses[-1]:.4f}")

    plot_training_history(train_losses, val_losses)
    print("training finished")
    evaluate_model(model, test_loader, DEVICE)

    print("Saving the trained model...")
    torch.save(model.state_dict(), "customCNN.pt")



if __name__ == "__main__":
    main()


