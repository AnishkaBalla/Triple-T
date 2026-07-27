import cv2
import numpy as np

#converting from CNN format to OpenCV format
def convert_box(box, image_width, image_height):
    cx, cy, w, h = box[:4]

    cx *= image_width
    cy *= image_height

    w *= image_width
    h*= image_height

    x1 = int(cx-w/2)
    y1 = int(cy-h/2)
    x2 = int(cx+w/2)
    y2 = int(cy+h/2)

    return x1, y1, x2, y2

#recieve image (cropped)
def crop_particle(image, bbox):
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    return crop

#detect contours inside one cropped particle
def find_particle_contours(crop):
    #convert RGB image to grayscale 
    gray = cv2.cvtColor(
        crop, cv2.COLOR_BGR2GRAY
    )
    #remove small noise
    blur = cv2.GaussianBlur(
        gray, (5,5), 0
    )

    #convert image to black and white
    _, threshold = cv2.threshold(
        blur, 20, 255, cv2.THRESH_BINARY
    )

    #find object outlines
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    return contours

# Main contour detection function

def detect_particles(image, predictions):
    height, width = image.shape[:2]
    results = []
    for index, prediction in enumerate(predictions):
        # ignore low confidence detections
        confidence = prediction[4]
        if confidence < 0.5:
            continue

        # convert CNN box to pixel coordinates
        bbox = convert_box(
            prediction,
            width,
            height
        )

        # crop detected particle
        crop = crop_particle(
            image,
            bbox
        )

        # find contours
        contours = find_particle_contours(crop)
        for contour in contours:
            # calculate contour area
            area = cv2.contourArea(contour)
            # ignore tiny noise
            if area < 10:
                continue
            # contour bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            # estimate particle length
            length = max(w, h)
            results.append({

                "particle_id": index,
                "confidence": float(confidence),
                "area": float(area),
                "bbox (location)": [
                    bbox[0] + x,
                    bbox[1] + y,
                    bbox[0] + x + w,
                    bbox[1] + y + h
                ],
                "fiber length": length

            })


    return results