import streamlit as st

#configure part
st.set_page_config(
    page_title = "Microplastic Detection", 
    layout = "wide" 
)

st.title("Microplastic Detection System")
st.write("Upload a microscopy image to locate microplastics using a custom CNN.")

#allows users to choose pics to upload 
upload_pic = st.file_uploader(
    "Choose a microscopy image", 
    type = ["png", "jpg", "jpeg"]
)

#if user has uploaded an image...
if upload_pic is not None:
    st.success("Image has been uploaded successfully! :)")
    st.image(upload_pic, caption = "Image Chosen", use_container_width = True)
    if st.button("Detect Microplastics"):
           st.info("Model will run here") #replace with actual prediction code by connecting model