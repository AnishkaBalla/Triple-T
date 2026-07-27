import streamlit as st
from PIL import Image, ImageDraw
import torch 
from torchvision import transforms 
from cnn_model import CustomCNN
from contour_detection import detect_particles
import numpy as np
import cv2
#configure part
st.set_page_config(
    page_title = "Microplastic Detection", 
    layout = "wide" 
)

st.title("Microplastic Detection System")
st.write("Upload a microscopy image to locate microplastics using a custom CNN.")

#load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN(max_objects = 28)
model.load_state_dict(torch.load("best_customCNN.pt", map_location = device))
model.to(device)
model.eval()

#PIL Image -> PyTorch Tensor
transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])

#allows users to choose pics to upload 
upload_pic = st.file_uploader(
    "Choose a microscopy image", 
    type = ["png", "jpg", "jpeg"]
)

#if user has uploaded an image...
if upload_pic is not None:
    st.success("Image has been uploaded successfully! :)")
    st.image(upload_pic, caption = "Image Chosen", width= 'stretch')
    if st.button("Detect Microplastics"):
        image = Image.open(upload_pic).convert("RGB")
        # prepare image for CNN
        input_tensor = transform(image).unsqueeze(0).to(device)
        # run model prediction
        with torch.no_grad():
            predictions = model(input_tensor)[0]
        # convert model outputs into usable values
        # boxes: cx, cy, width, height
        # confidence: probability
        predictions[:, 0:4] = torch.sigmoid(predictions[:, 0:4])
        predictions[:, 4] = torch.sigmoid(predictions[:, 4])
        # convert PIL image to OpenCV image
        image_cv = np.array(image)
        # run contour detection using your existing file
        contour_results = detect_particles(
            image_cv,
            predictions.cpu().numpy()
        )
        # draw detected particle contours/boxes
        annotated = image_cv.copy()
        for particle in contour_results:
            x1, y1, x2, y2 = particle["bbox (location)"]
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),   # yellow box
                2
            )
            cv2.putText(
                annotated,
                f"{particle['confidence']:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )
        # convert back to PIL because st.image works best with it
        annotated = Image.fromarray(annotated)
        st.success("Detection complete!")
        st.image(
            annotated,
            caption="Annotated Image",
            width= 'stretch'
        )