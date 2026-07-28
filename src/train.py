import subprocess  # import subprocess so the training script can install missing packages when needed.
import sys  # import sys so the script can access the current python executable for package installation.
from pathlib import Path  # import path so file paths can be handled in a portable way

import matplotlib  
matplotlib.use("Agg")  #non-interactive agg backend so plots can be saved in headless environments
import matplotlib.pyplot as plt  
import pandas as pd  
import torch  # import pytorch so the model, tensors, and training loop can be used.
import torch.nn.functional as F  # import functional operations so loss functions can be applied cleanly.
from PIL import Image  # import pil so images can be opened and converted into rgb.
from torch.utils.data import DataLoader, Dataset  #import dataloader and dataset so a custom dataset can be built for training.

# try to import the metrics and splitting utilities that are needed for evaluation.
# if the package is missing, install it automatically so the script can run on a fresh machine.
try:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score  
    from sklearn.model_selection import train_test_split  
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])  # install scikit-learn when it is not already present.
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score 
    from sklearn.model_selection import train_test_split  

from cnn_model import CustomCNN  #import the custom cnn architecture defined in the model file (cnn_model.py).
from dataloading_augmentation import dataset_path, train_transform  # import the selected image root and the shared image transform pipeline.
from checkpoint import log_results  # import the logging function to save training metrics to csv.

# this script trains a lightweight cnn to predict whether an image contains a microplastic object and where the box is located.
#it also saves simple plots so the training behavior is easy to inspect.
ROOT = Path(__file__).resolve().parent.parent  # resolve the repository root from the current file location.
IMAGES_DIRS = [
    dataset_path,
    ROOT / "new_data" / "augmented_train"
]  # store the image directory that should be used for training.
LABELS_PATH = ROOT / "new_data" / "archive" / "train" / "_annotations.csv"  # build the path to the annotation csv from the dataset root.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #choose the gpu if it is available, otherwise use the cpu.
BATCH_SIZE = 32  # set the batch size for mini-batch training.
EPOCHS = 70  # set the number of training epochs.
LEARNING_RATE = 0.001  # set the learning rate for the optimizer.
EVAL_CONF_THRESHOLD = 0.4  # use a slightly lower eval threshold to reduce false negatives on weak detections.
BOX_LOSS_WEIGHT = 5.0  # emphasize localization so predicted boxes align better with ground truth.
CONF_LOSS_WEIGHT = 0.5  # keep confidence loss important but not dominant over box quality.


#define a helper that creates a dataloader for a set of image paths and labels.
def make_loader(paths, labels_df, batch_size, shuffle):  # accept the paths, label table, batch size, and whether to shuffle.
    ds = BoxDataset(paths, labels_df, transform=train_transform)  # build a custom dataset object from the passed image paths and labels.
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)  # return a dataloader that batches the dataset for training.


#define a custom dataset that pairs each image with a target box and confidence label.
class BoxDataset(Dataset):  # inherit from pytorch's dataset class so it works with dataloader.
    def __init__(self, image_paths, labels_df, transform=None):  # initialize the dataset with image paths, labels, and an optional transform.
        self.image_paths = image_paths  # store the list of image paths for the dataset.
        self.labels_df = labels_df  #store the annotation dataframe that contains the target coordinates.
        self.transform = transform  # store the image transform pipeline.

    def __len__(self):  # return the number of images in the dataset.
        return len(self.image_paths)  # return the length of the stored image path list.

    def __getitem__(self, idx):  #load one image and its target label for a given index.
        path = self.image_paths[idx]  # get the image path at the requested index.
        image = Image.open(path).convert("RGB")  # open the image and convert it to rgb so it has three channels.
        MAX_OBJECTS = 28  #define the maximum number of microplastics that one image can contain. 
        # create an empty target tensor for 28 possible objects.
        # each object contains [x_center, y_center, width, height, confidence].
        target = torch.zeros((MAX_OBJECTS, 5), dtype=torch.float32)
        if self.transform:  # apply the transform pipeline (if provided)
            image = self.transform(image) 
        # look for all annotation rows matching this image file name.
        image_stem = path.stem.replace("_aug_0", "").replace("_aug_1", "")
        rows = self.labels_df[
            self.labels_df["filename"].apply(lambda x: Path(str(x)).stem) == image_stem
        ]
        rows = rows.sort_values(["xmin", "ymin", "xmax", "ymax"]).reset_index(drop=True)
        
        #loop through every microplastic annotation for this image.
        for i, (_, r) in enumerate(rows.iterrows()):

            # stop adding objects if the image contains more than 28 microplastics.
            if i >= MAX_OBJECTS:
                break

            # pull the box coordinates from the annotation row.
            x1, y1, x2, y2 = r["xmin"], r["ymin"], r["xmax"], r["ymax"]

            # pull the image width and height from the annotation row.
            width, height = r["width"], r["height"]

            #compute the normalized center x coordinate of the box.
            cx = ((x1 + x2) / 2) / width

            # compute the normalized center y coordinate of the box.
            cy = ((y1 + y2) / 2) / height

            # compute the normalized box width.
            bw = (x2 - x1) / width

            # compute the normalized box height.
            bh = (y2 - y1) / height

            #store this microplastic's box and confidence into the correct object slot.
            target[i] = torch.tensor(
                [cx, cy, bw, bh, 1.0],
                dtype=torch.float32
            )

        # return the processed image, all 28 possible object labels, and image path for visualization.
        return image, target, str(path)


# define a helper that plots the training, validation, and test loss curves.
def plot_training_history(train_losses, val_losses, test_losses, train_box_losses=None, train_conf_losses=None, val_box_losses=None, val_conf_losses=None, test_box_losses=None, test_conf_losses=None):  # accept the recorded training, validation, and test losses.
    plt.figure(figsize=(10, 6))  #create a new matplotlib figure with a larger size for more curves.
    plt.plot(train_losses, label="train loss", marker="o")  # plot the training loss values with markers.
    plt.plot(val_losses, label="validation loss", marker="o")  # plot the validation loss values with markers.
    plt.plot(test_losses, label="test loss", marker="o")  # plot the test loss values with markers.
    if train_box_losses is not None:
        plt.plot(train_box_losses, label="train box loss", marker="s", alpha=0.6)  #plot training bounding box loss.
    if val_box_losses is not None:
        plt.plot(val_box_losses, label="validation box loss", marker="s", alpha=0.6)  # plot validation bounding box loss.
    if test_box_losses is not None:
        plt.plot(test_box_losses, label="test box loss", marker="s", alpha=0.6)  # plot test bounding box loss.
    if train_conf_losses is not None:
        plt.plot(train_conf_losses, label="train conf loss", marker="^", alpha=0.6)  # plot training confidence loss.
    if val_conf_losses is not None:
        plt.plot(val_conf_losses, label="validation conf loss", marker="^", alpha=0.6)  #plot validation confidence loss.
    if test_conf_losses is not None:
        plt.plot(test_conf_losses, label="test conf loss", marker="^", alpha=0.6)  # plot test confidence loss.
    plt.xlabel("epoch")  # label the x-axis with the epoch number.
    plt.ylabel("loss")  # label the y-axis with loss.
    plt.title("training, validation, and test loss")  #give the plot a descriptive title.
    plt.legend()  # show the legend so each loss curve is identifiable.
    plt.grid(True, alpha=0.3)  # add a faint grid to improve readability.
    plt.tight_layout()  # adjust spacing so the plot is neatly arranged.
    plt.savefig("training_history.png")  #save the figure to a file in the current working directory.
    print("saved training curve to training_history.png")  # print a message to confirm the file was saved.
    plt.close()  # close the figure to free memory.


# define a helper that plots a confusion matrix for the binary presence/absence task.
def plot_confusion_matrix(y_true, y_pred):  #accept the true labels and predicted labels.
    labels = [0, 1]  # define the two class labels in the binary classification problem.
    cm = confusion_matrix(y_true, y_pred, labels=labels)  # compute the confusion matrix using the requested labels.
    class_names = ["background", "microplastic present"]  # define the human-readable class names for the matrix.
    plt.figure(figsize=(5, 4))  #create a compact figure for the confusion matrix.
    plt.imshow(cm, cmap="Blues")  # draw the matrix
    plt.xticks(labels, class_names)  # x-axis ticks as the class names.
    plt.yticks(labels, class_names)  # y-axis ticks as the class names.
    plt.xlabel("predicted class")  #x-axis as predicted class.
    plt.ylabel("true class")  # y-axis as true class.
    plt.title("confusion matrix for microplastic presence")  
    for i in range(cm.shape[0]):  # loop over the rows of the matrix.
        for j in range(cm.shape[1]):  # loop over the columns of the matrix.
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")  #add the count into the heatmap cell.
    plt.tight_layout()  # adjust the layout so the labels fit neatly inside the figure
    plt.savefig("confusion_matrix.png")  # save the confusion matrix figure to disk
    print("saved confusion matrix to confusion_matrix.png")  
    plt.close()  

def convert_norm_to_corners(box):
    # converting [cx, cy, w, h] to [xmin, ymin, xmax, ymax]
    cx, cy,w,h = box
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2

    return [xmin, ymin, xmax, ymax]

#calculate iou - box1 and box2 should both be in [xmin, ymin...] format
def calculate_iou(box1, box2):
    # clamp all coordinates into valid normalized image bounds before computing overlap.
    box1 = [max(0.0, min(1.0, v)) for v in box1]
    box2 = [max(0.0, min(1.0, v)) for v in box2]

    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    intersection = max(0, xB - xA) * max(0, yB - yA)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union # area of overlap over union

# define a helper that evaluates the trained model using common classification metrics.
def evaluate_model(model, loader, device):  # accept the trained model, evaluation loader, and device.
    model.eval()  #put the model into evaluation mode so dropout and batch norm behave predictably.

    all_true = []  # empty list to collect true labels from all batches.
    all_pred = []  # create an empty list to collect predicted labels from all batches.
    ious = [] # empty list of iou
    iou_threshold = 0.3  # set IoU threshold for detection matching; slightly lower to avoid all-zero metrics early in training.
    conf_threshold = EVAL_CONF_THRESHOLD  # evaluate detections with the configured threshold.
    matched_ious = []  # IoUs from one-to-one matched detection pairs.

    # keep explicit detection counts so localization mistakes penalize both precision and recall.
    tp = 0
    fp = 0
    fn = 0

    with torch.no_grad():  #disable gradient tracking during evaluation to save memory.
        for images, targets, _ in loader:  # loop over each batch from the validation loader.

            images = images.to(device)  # move the input images to the selected device.

            preds = model(images)  # get all 28 predictions from the model.

            pred_boxes = torch.sigmoid(preds[:, :, :4]).cpu() #extract normalized predicted boxes for all 28 objects.
            pred_conf = preds[:, :, 4].cpu() # extract confidence logits for all 28 objects.

            target_boxes = targets[:, :, :4] # extract target boxes for all 28 objects.
            target_conf = targets[:, :, 4] # extract target confidence for all 28 objects.

            # classification probabilities per object slot
            probs = torch.sigmoid(pred_conf)

            #calculate iou and detection matches per image using one-to-one matching.
            for batch in range(preds.shape[0]):
                gt_indices = (target_conf[batch] >= 0.5).nonzero(as_tuple=False).flatten().tolist()
                pred_indices_all = list(range(pred_boxes.shape[1]))
                pred_indices_det = (probs[batch] >= conf_threshold).nonzero(as_tuple=False).flatten().tolist()

                gt_xyxy = [
                    convert_norm_to_corners(target_boxes[batch][g].tolist())
                    for g in gt_indices
                ]
                pred_xyxy_all = [
                    convert_norm_to_corners(pred_boxes[batch][p].tolist())
                    for p in pred_indices_all
                ]
                pred_xyxy_det = [
                    convert_norm_to_corners(pred_boxes[batch][p].tolist())
                    for p in pred_indices_det
                ]

                # localization quality: for each GT, use best IoU across all predicted slots.
                for gt_box in gt_xyxy:
                    if len(pred_xyxy_all) == 0:
                        ious.append(0.0)
                        continue
                    best_iou = max(calculate_iou(pred_box, gt_box) for pred_box in pred_xyxy_all)
                    ious.append(best_iou)

                # detection quality: greedy one-to-one matching for detections above threshold.
                candidate_pairs = []
                for p_i, pred_box in enumerate(pred_xyxy_det):
                    for g_i, gt_box in enumerate(gt_xyxy):
                        pair_iou = calculate_iou(pred_box, gt_box)
                        if pair_iou >= iou_threshold:
                            candidate_pairs.append((pair_iou, p_i, g_i))

                candidate_pairs.sort(key=lambda x: x[0], reverse=True)
                matched_pred = set()
                matched_gt = set()
                for _, p_i, g_i in candidate_pairs:
                    if p_i in matched_pred or g_i in matched_gt:
                        continue
                    matched_pred.add(p_i)
                    matched_gt.add(g_i)
                    matched_ious.append(calculate_iou(pred_xyxy_det[p_i], gt_xyxy[g_i]))

                image_tp = len(matched_gt)
                image_fp = len(pred_xyxy_det) - image_tp
                image_fn = len(gt_xyxy) - image_tp
                tp += image_tp
                fp += image_fp
                fn += image_fn

                # keep slot-level labels for confusion matrix plotting.
                for obj in range(28):
                    gt_exists = target_conf[batch][obj] >= 0.5
                    pred_exists = probs[batch][obj] >= conf_threshold
                    all_true.append(int(gt_exists))
                    all_pred.append(int(pred_exists))


    y_true = all_true  # all true slot labels -> one list.
    y_pred = all_pred  # all predicted slot labels -> one list.

    total = tp + fp + fn
    acc = tp / total if total else 0.0  # IoU-aware detection accuracy
    prec = tp / (tp + fp) if (tp + fp) else 0.0  # IoU-aware precision
    rec = tp / (tp + fn) if (tp + fn) else 0.0  # IoU-aware recall
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0  # IoU-aware f1

    # also report raw confidence-only slot metrics for troubleshooting threshold behavior.
    raw_acc = accuracy_score(y_true, y_pred)  # raw slot accuracy (ignores IoU)
    raw_prec = precision_score(y_true, y_pred, zero_division=0)  # raw slot precision
    raw_rec = recall_score(y_true, y_pred, zero_division=0)  # raw slot recall
    raw_f1 = f1_score(y_true, y_pred, zero_division=0)  # raw slot f1


    print("accuracy:", round(raw_acc, 4))  # rounded slot-classification accuracy.
    print("precision:", round(raw_prec, 4))  #rounded slot-classification precision.
    print("recall:", round(raw_rec, 4))  # rounded slot-classification recall.
    print("f1 score:", round(raw_f1, 4))  # rounded slot-classification f1 score.
    print("iou-aware detection accuracy:", round(acc, 4))
    print("iou-aware detection precision:", round(prec, 4))
    print("iou-aware detection recall:", round(rec, 4))
    print("iou-aware detection f1:", round(f1, 4))
    print("raw slot accuracy:", round(raw_acc, 4))
    print("raw slot precision:", round(raw_prec, 4))
    print("raw slot recall:", round(raw_rec, 4))
    print("raw slot f1:", round(raw_f1, 4))


    if len(matched_ious)>0:
        print("Mean IoU:", f"{(sum(matched_ious) / len(matched_ious)):.6f}")
    elif len(ious)>0:
        print("Mean IoU:", f"{(sum(ious) / len(ious)):.6f}")
    else:
        print("No IoUs")


    plot_confusion_matrix(y_true, y_pred)  # plot the confusion matrix using the aggregated labels.
#visualization!
def visualize_predictions(model, loader, device):
    model.eval()

    with torch.no_grad(): # disable gradients to make sure you dont update the model

        for images, targets, paths in loader:
            preds = model(images.to(device))
            preds = preds[0].cpu()   # all 28 predictions for first image
            preds[:,0:4] = torch.sigmoid(preds[:,0:4])
            preds[:,4] = torch.sigmoid(preds[:,4])
            all_preds = preds.clone()


            target = targets[0]
            path = paths[0]

            confidence_threshold = EVAL_CONF_THRESHOLD
            preds = preds[preds[:,4] > confidence_threshold]

            # if nothing passes threshold, keep top few predictions so localization can still be inspected.
            if preds.shape[0] == 0:
                fallback_k = 3
                top_idx = torch.argsort(all_preds[:,4], descending=True)[:fallback_k]
                preds = all_preds[top_idx]

            # showing image
            image = Image.open(path).convert("RGB")
            fig, ax = plt.subplots(figsize = (8,8))
            ax.imshow(image)

            #drawing boxes (green being ground, red being predicted)
            import matplotlib.patches as patches 
            for box in target:
                if box[4] == 0:
                    continue
                xmin, ymin, xmax, ymax = convert_norm_to_corners(box[:4].tolist())
                w, h = image.size
                xmin *= w
                xmax *= w
                ymin *= h
                ymax *= h

                rect = patches.Rectangle(
                    (xmin, ymin), 
                    xmax - xmin, ymax - ymin, 
                    edgecolor = "green", 
                    facecolor = "none", 
                    linewidth = 2
                )
                ax.add_patch(rect)
            for box in preds:
                if box[4] == 0:
                    continue
                xmin, ymin, xmax, ymax = convert_norm_to_corners(box[:4].tolist())
                w, h = image.size
                xmin *= w
                xmax *= w
                ymin *= h
                ymax *= h

                rect = patches.Rectangle(
                    (xmin, ymin), 
                    xmax - xmin, ymax - ymin, 
                    edgecolor = "red", 
                    facecolor = "none", 
                    linewidth = 2
                )
                ax.add_patch(rect)
            plt.savefig("prediction_example.png")
            plt.close()


# define the main training workflow.
def main():  # trianing pipeline
    torch.manual_seed(42)  # set a fixed random seed so training is reproducible.
    labels_df = pd.read_csv(LABELS_PATH)  #load the annotation dataframe from the labels csv.
    image_paths = []

    for folder in IMAGES_DIRS:
        image_paths.extend(
            [
                p for p in folder.rglob("*")
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )

    print("total images:", len(image_paths))  # collect all image paths from the chosen image root.
    train_files, test_files = train_test_split(image_paths, test_size=0.3, random_state=42)  # split the image paths into train and test sets.
    train_loader = make_loader(train_files, labels_df, BATCH_SIZE, shuffle=True)  # create the training dataloader.
    test_loader = make_loader(test_files, labels_df, BATCH_SIZE, shuffle=False)  #create the test dataloader.
    MAX_OBJECTS = 28
    model = CustomCNN(max_objects=MAX_OBJECTS).to(DEVICE)  # create the cnn model and move it to the selected device.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # create the adam optimizer for training.
    box_loss_fn = torch.nn.SmoothL1Loss()  # create the loss function for bounding box regression.
    conf_loss_fn = torch.nn.BCEWithLogitsLoss()  #create the loss function for the presence confidence.

    train_losses = []  # create an empty list to record training losses per epoch.
    val_losses = []  # create an empty list to record validation losses per epoch.
    test_losses = []  # create an empty list to record test losses per epoch.
    train_box_losses = []  #create an empty list to record training bounding box losses per epoch.
    train_conf_losses = []  # create an empty list to record training confidence losses per epoch.
    val_box_losses = []  # create an empty list to record validation bounding box losses per epoch.
    val_conf_losses = []  # create an empty list to record validation confidence losses per epoch.
    test_box_losses = []  #create an empty list to record test bounding box losses per epoch.
    test_conf_losses = []  # create an empty list to record test confidence losses per epoch.

    for epoch in range(EPOCHS):  # loop over the configured number of epochs.
        model.train()  # set the model to training mode so dropout is active.
        train_loss = 0.0  #reset the cumulative training loss for the epoch.
        train_box_loss_epoch = 0.0  # reset the cumulative training bounding box loss for the epoch.
        train_conf_loss_epoch = 0.0  # reset the cumulative training confidence loss for the epoch.
        for images, targets, _ in train_loader:  # loop over each training batch.
            images, targets = images.to(DEVICE), targets.to(DEVICE)  #move the batch to the selected device.
            optimizer.zero_grad()  # clear old gradients before the backward pass.
            preds = model(images)
            pred_boxes = torch.sigmoid(preds[:, :, :4])
            pred_conf_logits = preds[:, :, 4]
            target_boxes = targets[:, :, :4]
            target_conf = targets[:, :, 4]
            box_loss_raw = F.smooth_l1_loss(pred_boxes, target_boxes, reduction="none")  # compute elementwise box regression loss for each sample.
            positive_slots = target_conf.sum().clamp(min=1.0)
            box_loss = (box_loss_raw.mean(dim=2) * target_conf).sum() / positive_slots  # normalize box loss by positive slots so localization gradients are strong.
            pos_weight = ((target_conf.numel() - target_conf.sum()) / positive_slots).detach().clamp(min=1.0, max=20.0)
            conf_loss = F.binary_cross_entropy_with_logits(pred_conf_logits, target_conf, pos_weight=pos_weight)  #compute confidence loss with positive-class reweighting.
            loss = (CONF_LOSS_WEIGHT * conf_loss) + (BOX_LOSS_WEIGHT * box_loss)  # combine losses with stronger localization emphasis.

            loss.backward()  # backpropagate the training loss through the network.
            optimizer.step()  # update the model parameters using the optimizer.
            train_loss += loss.item()  #add the current batch loss to the epoch total.
            train_box_loss_epoch += box_loss.item()  # add the current batch bounding box loss to the epoch total.
            train_conf_loss_epoch += conf_loss.item()  # add the current batch confidence loss to the epoch total.

        model.eval()  # switch the model back to evaluation mode for validation.
        val_loss = 0.0  #reset the cumulative validation loss for the epoch.
        val_box_loss_epoch = 0.0  # reset the cumulative validation bounding box loss for the epoch.
        val_conf_loss_epoch = 0.0  # reset the cumulative validation confidence loss for the epoch.
        test_loss = 0.0  # reset the cumulative test loss for the epoch.
        test_box_loss_epoch = 0.0  #reset the cumulative test bounding box loss for the epoch.
        test_conf_loss_epoch = 0.0  # reset the cumulative test confidence loss for the epoch.
        with torch.no_grad():  # disable gradient tracking during validation.
            for images, targets, _ in test_loader:  # loop over each validation and test batch.
                images, targets = images.to(DEVICE), targets.to(DEVICE)  #move the batch to the selected device.
                preds = model(images)  # run the images through the model and get all 28 object predictions.
                pred_boxes = torch.sigmoid(preds[:, :, :4])  # extract normalized bounding boxes for all 28 predictions.
                pred_conf_logits = preds[:, :, 4]  # extract confidence values for all 28 predictions.
                target_boxes = targets[:, :, :4]  #extract ground truth boxes for all 28 objects.
                target_conf = targets[:, :, 4]  # extract confidence labels for all 28 objects.
                box_loss_raw = F.smooth_l1_loss(pred_boxes,target_boxes,reduction="none")

                # average x,y,w,h errors for each object
                positive_slots = target_conf.sum().clamp(min=1.0)
                box_loss = (box_loss_raw.mean(dim=2) * target_conf).sum() / positive_slots

                pos_weight = ((target_conf.numel() - target_conf.sum()) / positive_slots).detach().clamp(min=1.0, max=20.0)
                conf_loss = F.binary_cross_entropy_with_logits(pred_conf_logits, target_conf, pos_weight=pos_weight)  # compute the confidence loss for the validation data.
                loss = (CONF_LOSS_WEIGHT * conf_loss) + (BOX_LOSS_WEIGHT * box_loss)  #combine losses with stronger localization emphasis for validation.
                val_loss += loss.item()  # add the current batch loss to the validation total.
                val_box_loss_epoch += box_loss.item()  # add the current batch bounding box loss to the validation total.
                val_conf_loss_epoch += conf_loss.item()  # add the current batch confidence loss to the validation total.
                test_loss += loss.item()  #add the current batch loss to the test total.
                test_box_loss_epoch += box_loss.item()  # add the current batch bounding box loss to the test total.
                test_conf_loss_epoch += conf_loss.item()  # add the current batch confidence loss to the test total.

        train_losses.append(train_loss / len(train_loader))  # record the average training loss for the epoch.
        val_losses.append(val_loss / len(test_loader))  #record the average validation loss for the epoch.
        test_losses.append(test_loss / len(test_loader))  # record the average test loss for the epoch.
        train_box_losses.append(train_box_loss_epoch / len(train_loader))  # record the average training bounding box loss for the epoch.
        train_conf_losses.append(train_conf_loss_epoch / len(train_loader))  # record the average training confidence loss for the epoch.
        val_box_losses.append(val_box_loss_epoch / len(test_loader))  #record the average validation bounding box loss for the epoch.
        val_conf_losses.append(val_conf_loss_epoch / len(test_loader))  # record the average validation confidence loss for the epoch.
        test_box_losses.append(test_box_loss_epoch / len(test_loader))  # record the average test bounding box loss for the epoch.
        test_conf_losses.append(test_conf_loss_epoch / len(test_loader))  # record the average test confidence loss for the epoch.
        log_results(epoch + 1, train_losses[-1], val_losses[-1], test_losses[-1], LEARNING_RATE, train_box_losses[-1], train_conf_losses[-1], val_box_losses[-1], val_conf_losses[-1], test_box_losses[-1], test_conf_losses[-1])  #log the epoch metrics to csv.
        print(f"epoch {epoch + 1}/{EPOCHS} | train loss: {train_losses[-1]:.4f} | val loss: {val_losses[-1]:.4f} | test loss: {test_losses[-1]:.4f}")  # print the epoch summary.

    plot_training_history(train_losses, val_losses, test_losses, train_box_losses, train_conf_losses, val_box_losses, val_conf_losses, test_box_losses, test_conf_losses)  # plot the saved training, validation, and test loss curves.
    print("training finished")  # print a completion message for the overall training loop.
    evaluate_model(model, test_loader, DEVICE)  #evaluate the trained model on the test set.
    visualize_predictions(model, test_loader, DEVICE) # visualize boxes
    print("Saving the trained model...")  # print a message before saving the model weights.
    if val_losses[-1] == min(val_losses):
        torch.save(model.state_dict(), "best_customCNN.pt")  # save the best trained model state dictionary to a file.

    
    

#run the training workflow when the script is executed directly.
if __name__ == "__main__":  # check whether the file is being run as a script.
    main()  # start the main training workflow.


