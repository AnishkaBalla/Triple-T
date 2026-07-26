import torch
from torchvision import transforms
from PIL import Image
from cnn_model import CustomCNN


# loading trained cnn model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomCNN(max_objects=50)
model.load_state_dict(torch.load('customCNN.pt', map_location=device))
model.to(device) # moves the model to the device (gpu or cpu) for inference
model.eval() # notifies network that we r using this for testing


#image preprocessing (resize to 256x256 and convert to tensor)
transforms_inference = transforms.Compose([
    transforms.Resize((256, 256)), # resizing image to 256x256 since that was the size the images were during training
    transforms.ToTensor(),  # makes pixel values go from 0-255 to 0-1 and converts to tensor
])


def run_inference(image_path, conf_threshold=0.5):
    original_image = Image.open(image_path).convert("RGB") # convert to rgb since we trained the model on rgb images
    input_tensor = transforms_inference(original_image).unsqueeze(0).to(device) #allows dimension to be a batch, for example [1, 3, 256, 256] instead of [3, 256, 256] (1 represents batch size)
   
    with torch.no_grad(): # makes sure torch dont run gradients since we not training rn
        outputs = model(input_tensor)
        pred_boxes = outputs['boxes'][0].cpu()   # shape: [n, 4]
        pred_scores = outputs['scores'][0].cpu() # shape: [n]
       
    #removing low-confidence detections based on the threshold
    valid_indices = pred_scores >= conf_threshold # filters out low-confidence detections
    filtered_boxes = pred_boxes[valid_indices]
    filtered_scores = pred_scores[valid_indices]
   
    # counting the number of detected microplastics
    num_microplastics = len(filtered_scores)
   
    # convert normalized coordinates to pixel coordinates
    pixel_boxes = []
    for box in filtered_boxes:
        x1, y1, x2, y2 = box.tolist() #extracting the coordinates from line 28
        px_x1 = x1 * 256
        px_y1 = y1 * 256
        px_x2 = x2 * 256
        px_y2 = y2 * 256
        pixel_boxes.append([px_x1, px_y1, px_x2, px_y2])
       
    return {
        "num_detections": num_microplastics,
        "pixel_boxes": pixel_boxes,
        "confidence_scores": filtered_scores.tolist()
    }


# example test on random testing image, make sure to replace "random_testing_image.jpg" with the actual path to the test image
result = run_inference("data\microplastic-dataset-for-computer-vision\organized_images\ClassB\b--18-_jpg.rf.47619bec35dba029d567c6097eac49de.jpg", conf_threshold=0.7)
print(f"Detected microplastics: {result['num_detections']}")
print(f"Pixel Coordinates: {result['pixel_boxes']}")
print(f"Confidence Scores: {result['confidence_scores']}")



