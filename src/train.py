import subprocess  #import subprocess so the training script can install missing packages when needed.
import sys  #import sys so the script can access the current Python executable for package installation.
from pathlib import Path  #import Path so file paths can be handled in a portable way

import matplotlib  
matplotlib.use("Agg")  #non-interactive Agg backend so plots can be saved in headless environments
import matplotlib.pyplot as plt  
import pandas as pd  
import torch  #import PyTorch so the model, tensors, and training loop can be used.
import torch.nn.functional as F  #import functional operations so loss functions can be applied cleanly.
from PIL import Image  #import PIL so images can be opened and converted into RGB.
from torch.utils.data import DataLoader, Dataset  #import DataLoader and Dataset so a custom dataset can be built for training.

#try to import the metrics and splitting utilities that are needed for evaluation.
# if the package is missing, install it automatically so the script can run on a fresh machine.
try:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score  
    from sklearn.model_selection import train_test_split  
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])  # install scikit-learn when it is not already present.
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score 
    from sklearn.model_selection import train_test_split  

from cnn_model import CustomCNN  #import the custom CNN architecture defined in the model file (cnn_model.py).
from dataloading_augmentation import dataset_path, train_transform  #import the selected image root and the shared image transform pipeline.

# this script trains a lightweight CNN to predict whether an image contains a microplastic object and where the box is located.
#it also saves simple plots so the training behavior is easy to inspect.
ROOT = Path(__file__).resolve().parent.parent  # resolve the repository root from the current file location.
IMAGES_DIR = dataset_path  #store the image directory that should be used for training.
LABELS_PATH = dataset_path / "dataset_manifest.csv"  # build the path to the manifest CSV from the selected image folder.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # choose the GPU if it is available, otherwise use the CPU.
BATCH_SIZE = 32  # set the batch size for mini-batch training.
EPOCHS = 2  # set the number of training epochs.
LEARNING_RATE = 0.001  # set the learning rate for the optimizer.


# define a helper that creates a DataLoader for a set of image paths and labels.
def make_loader(paths, labels_df, batch_size, shuffle):  # Accept the paths, label table, batch size, and whether to shuffle.
    ds = BoxDataset(paths, labels_df, transform=train_transform)  # Build a custom dataset object from the passed image paths and labels.
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)  # Return a DataLoader that batches the dataset for training.


# Define a custom dataset that pairs each image with a target box and confidence label.
class BoxDataset(Dataset):  # inherit from PyTorch's Dataset class so it works with DataLoader.
    def __init__(self, image_paths, labels_df, transform=None):  # initialize the dataset with image paths, labels, and an optional transform.
        self.image_paths = image_paths  # store the list of image paths for the dataset.
        self.labels_df = labels_df  # store the annotation DataFrame that contains the target coordinates.
        self.transform = transform  # store the image transform pipeline.

    def __len__(self):  #return the number of images in the dataset.
        return len(self.image_paths)  # return the length of the stored image path list.

    def __getitem__(self, idx):  # load one image and its target label for a given index.
        path = self.image_paths[idx]  # get the image path at the requested index.
        image = Image.open(path).convert("RGB")  # open the image and convert it to RGB so it has three channels.
        if self.transform:  # apply the transform pipeline (if provided)
            image = self.transform(image)  

        row = self.labels_df[self.labels_df["filename"] == path.name]  # look for any annotation rows matching this image file name.
        if row.empty:  # if there is no annotation for this image, treat it as a background example.
            target = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)  # create an empty target vector (similar to YOLO!) with zeroed box coordinates and confidence.
        else:  # If there is an annotation row for the image, convert it into a normalized target box.
            r = row.iloc[0]  # grab the first matching row for the image.
            x1, y1, x2, y2 = r["xmin"], r["ymin"], r["xmax"], r["ymax"]  # pull the box coordinates from the annotation row.
            width, height = r["width"], r["height"]  # pull the image width and height from the annotation row.
            cx = ((x1 + x2) / 2) / width  # compute the normalized center x coordinate of the box.
            cy = ((y1 + y2) / 2) / height  #compute the normalized center y coordinate of the box.
            bw = (x2 - x1) / width  # compute the normalized box width.
            bh = (y2 - y1) / height  #compute the normalized box height.
            target = torch.tensor([cx, cy, bw, bh, 1.0], dtype=torch.float32)  #create a target tensor with the normalized box and confidence 1.

        return image, target  # Return the processed image and target tensor.


# Define a helper that plots the training and validation loss curves.
def plot_training_history(train_losses, val_losses):  # Accept the recorded training and validation losses.
    plt.figure(figsize=(6, 4))  # Create a new matplotlib figure with a moderate size.
    plt.plot(train_losses, label="train loss", marker="o")  # Plot the training loss values with markers.
    plt.plot(val_losses, label="validation loss", marker="o")  # Plot the validation loss values with markers.
    plt.xlabel("epoch")  # Label the x-axis with the epoch number.
    plt.ylabel("loss")  # Label the y-axis with loss.
    plt.title("training and validation loss")  # Give the plot a descriptive title.
    plt.legend()  # Show the legend so each loss curve is identifiable.
    plt.grid(True, alpha=0.3)  # Add a faint grid to improve readability.
    plt.tight_layout()  # Adjust spacing so the plot is neatly arranged.
    plt.savefig("training_history.png")  # Save the figure to a file in the current working directory.
    print("saved training curve to training_history.png")  # Print a message to confirm the file was saved.
    plt.close()  # Close the figure to free memory.


# Define a helper that plots a confusion matrix for the binary presence/absence task.
def plot_confusion_matrix(y_true, y_pred):  # Accept the true labels and predicted labels.
    labels = [0, 1]  # Define the two class labels in the binary classification problem.
    cm = confusion_matrix(y_true, y_pred, labels=labels)  # Compute the confusion matrix using the requested labels.
    class_names = ["background", "microplastic present"]  # Define the human-readable class names for the matrix.
    plt.figure(figsize=(5, 4))  #create a compact figure for the confusion matrix.
    plt.imshow(cm, cmap="Blues")  # draw the matrix 
    plt.xticks(labels, class_names)  # x-axis ticks as the class names.
    plt.yticks(labels, class_names)  # y-axis ticks as the class names.
    plt.xlabel("predicted class")  # x-axis as predicted class.
    plt.ylabel("true class")  # y-axis as true class.
    plt.title("confusion matrix for microplastic presence")  
    for i in range(cm.shape[0]):  # loop over the rows of the matrix.
        for j in range(cm.shape[1]):  # Loop over the columns of the matrix.
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")  # add the count into the heatmap cell.
    plt.tight_layout()  # adjust the layout so the labels fit neatly inside the figure
    plt.savefig("confusion_matrix.png")  # save the confusion matrix figure to disk
    print("saved confusion matrix to confusion_matrix.png")  
    plt.close()  


#define a helper that evaluates the trained model using common classification metrics.
def evaluate_model(model, loader, device):  # Accept the trained model, evaluation loader, and device.
    model.eval()  #put the model into evaluation mode so dropout and batch norm behave predictably.
    all_true = []  #empty list to collect true labels from all batches.
    all_pred = []  # create an empty list to collect predicted labels from all batches.
    with torch.no_grad():  # disable gradient tracking during evaluation to save memory.
        for images, targets in loader:  #loop over each batch from the validation loader.
            images = images.to(device)  # move the input images to the selected device.
            preds = model(images)[:, 0, :]  # extract the model prediction for the first object slot.
            probs = torch.sigmoid(preds[:, 4]).cpu()  # convert the confidence logit -> probability.
            pred_labels = (probs >= 0.5).long()  # Convert the probability -> binary label.
            true_labels = (targets[:, 4] >= 0.5).long().cpu()  # Convert the target confidence into a binary label.
            all_true.append(true_labels)  #  true labels for later aggregation.
            all_pred.append(pred_labels)  # predicted labels for later aggregation.

    y_true = torch.cat(all_true).numpy()  #all true labels -> one array.
    y_pred = torch.cat(all_pred).numpy()  # all predicted labels ->one array.
    acc = accuracy_score(y_true, y_pred)  # accuracy
    prec = precision_score(y_true, y_pred, zero_division=0)  # precision 
    rec = recall_score(y_true, y_pred, zero_division=0)  # recall
    f1 = f1_score(y_true, y_pred, zero_division=0)  # F1 score

    print("accuracy:", round(acc, 4))  # rounded accuracy score.
    print("precision:", round(prec, 4))  #rounded precision score.
    print("recall:", round(rec, 4))  # rounded recall score.
    print("f1 score:", round(f1, 4))  #rounded F1 score.
    plot_confusion_matrix(y_true, y_pred)  #plot the confusion matrix using the aggregated labels.


# define the main training workflow.
def main():  #trianing pipeline
    torch.manual_seed(42)  # set a fixed random seed so training is reproducible.
    labels_df = pd.read_csv(LABELS_PATH)  # load the annotation DataFrame from the labels CSV.
    image_paths = [p for p in IMAGES_DIR.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]  # Collect all image paths from the chosen image root.
    train_files, test_files = train_test_split(image_paths, test_size=0.3, random_state=42)  # Split the image paths into train and test sets.
    train_loader = make_loader(train_files, labels_df, BATCH_SIZE, shuffle=True)  # Create the training DataLoader.
    test_loader = make_loader(test_files, labels_df, BATCH_SIZE, shuffle=False)  # Create the test DataLoader.

    model = CustomCNN(max_objects=1).to(DEVICE)  # Create the CNN model and move it to the selected device.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Create the Adam optimizer for training.
    box_loss_fn = torch.nn.SmoothL1Loss()  # Create the loss function for bounding box regression.
    conf_loss_fn = torch.nn.BCEWithLogitsLoss()  # Create the loss function for the presence confidence.

    train_losses = []  # Create an empty list to record training losses per epoch.
    val_losses = []  #create an empty list to record validation losses per epoch.

    for epoch in range(EPOCHS):  # Loop over the configured number of epochs.
        model.train()  # Set the model to training mode so dropout is active.
        train_loss = 0.0  # Reset the cumulative training loss for the epoch.
        for images, targets in train_loader:  # Loop over each training batch.
            images, targets = images.to(DEVICE), targets.to(DEVICE)  # Move the batch to the selected device.
            optimizer.zero_grad()  # Clear old gradients before the backward pass.
            preds = model(images)[:, 0, :]  #Run the images through the model and take the first object prediction.
            pred_boxes = preds[:, :4]  # Extract the predicted box coordinates from the model output.
            pred_conf_logits = preds[:, 4]  #Extract the predicted confidence logit from the model output.
            target_boxes = targets[:, :4]  # Extract the target box coordinates from the labels.
            target_conf = targets[:, 4]  # Extract the target confidence value from the labels.

            box_loss_raw = F.smooth_l1_loss(pred_boxes, target_boxes, reduction="none").mean(dim=1)  # Compute elementwise box regression loss for each sample.
            box_loss = (box_loss_raw * target_conf).mean()  # Weight the box loss by whether the sample actually contains an object.
            conf_loss = conf_loss_fn(pred_conf_logits, target_conf)  # Compute the binary confidence loss.
            loss = conf_loss + box_loss  # Combine the confidence loss and box loss into one overall loss.

            loss.backward()  # Backpropagate the training loss through the network.
            optimizer.step()  #Update the model parameters using the optimizer.
            train_loss += loss.item()  # Add the current batch loss to the epoch total.

        model.eval()  # Switch the model back to evaluation mode for validation.
        val_loss = 0.0  # Reset the cumulative validation loss for the epoch.
        with torch.no_grad():  # Disable gradient tracking during validation.
            for images, targets in test_loader:  # Loop over each validation batch.
                images, targets = images.to(DEVICE), targets.to(DEVICE)  # Move the validation batch to the selected device.
                preds = model(images)[:, 0, :]  # Run the validation images through the model.
                pred_boxes = preds[:, :4]  # Extract the predicted box coordinates from the model output.
                pred_conf_logits = preds[:, 4]  # Extract the predicted confidence logit from the model output.
                target_boxes = targets[:, :4]  # Extract the target box coordinates from the labels.
                target_conf = targets[:, 4]  # Extract the target confidence value from the labels.

                box_loss_raw = F.smooth_l1_loss(pred_boxes, target_boxes, reduction="none").mean(dim=1)  # Compute the box loss again for the validation data.
                box_loss = (box_loss_raw * target_conf).mean()  # Weight the box loss by the presence label.
                conf_loss = conf_loss_fn(pred_conf_logits, target_conf)  # Compute the confidence loss for the validation data.
                loss = conf_loss + box_loss  # Combine the confidence and box losses for validation.
                val_loss += loss.item()  # Add the current validation batch loss to the epoch total.

        train_losses.append(train_loss / len(train_loader))  # Record the average training loss for the epoch.
        val_losses.append(val_loss / len(test_loader))  # Record the average validation loss for the epoch.
        print(f"epoch {epoch + 1}/{EPOCHS} | train loss: {train_losses[-1]:.4f} | validation loss: {val_losses[-1]:.4f}")  # Print the epoch summary.

    plot_training_history(train_losses, val_losses)  # Plot the saved training and validation loss curves.
    print("training finished")  #Print a completion message for the overall training loop.
    evaluate_model(model, test_loader, DEVICE)  # Evaluate the trained model on the test set.

    print("Saving the trained model...")  # Print a message before saving the model weights.
    torch.save(model.state_dict(), "customCNN.pt")  # Save the trained model state dictionary to a file.


#Run the training workflow when the script is executed directly.
if __name__ == "__main__":  # Check whether the file is being run as a script.
    main()  # Start the main training workflow.


