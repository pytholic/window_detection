# Import modules

import os
import ffmpeg
import time
import cv2
import numpy as np
import subprocess as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import albumentations as A

from albumentations.pytorch import ToTensorV2
from PIL import Image
from matplotlib import pyplot as plt

# Set device
device = torch.device("cuda")  # "cuda:0"
print(device)


# Define parameters
workdir = os.getcwd()
#video_path = "/home/pytholic/Desktop/Projects/icms_data/test_videos/test5.mp4"
video_path = '/home/pytholic/Desktop/Projects/icms_data/test_videos/new_car/0099.mp4'
model_path = workdir + "/model/model_new/model_densenet.pth"
out_filename = workdir + "/frames/out.avi"
NUM_CLASSES = 2
classes = ["open", "close"]

four_k = (3840, 2160)
FULL_HD = (1920,1080)
SD = (640, 480)

### Utility functions ###

# Utility to apply transforms
def get_transform():
    resize = A.Resize(224, 224)
    normalize = A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    to_tensor = ToTensorV2()
    return A.Compose([resize, normalize, to_tensor])

# Print function
def print_text(
    img,
    text: str,
    org=(100, 100),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=1,
    color=(0, 255, 0),
    thickness=2,
):
    cv2.putText(
        img,
        text,
        org=org,
        fontFace=fontFace,
        fontScale=fontScale,
        color=color,
        thickness=thickness,
    )


### Prediction ###

# Classification function
def classify(model, image_transforms, img, classes):
    img = image_transforms(image = img)["image"]
    img = img.unsqueeze(0)
    img = img.to(device)
    #img = img.half()

    output = model(img)
    _, prediction = torch.max(output.data, 1)
    predicted_class = classes[prediction.item()]

    return predicted_class

def predict(video_path, out_filename, model, transforms, write=False):

    if write:

        ext = str(out_filename.split("/")[-1].split(".")[-1])
        if ext == "mp4":
            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        else:
            fourcc = cv2.VideoWriter_fourcc("x", "v", "i", "d")

        out = cv2.VideoWriter(out_filename, fourcc, 30.0, (WIDTH, HEIGHT))
    
    #cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(2)
    res = load_state_dict                                                                                                                                                                                                                                                               
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    print(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    while True:

        ### INPUT ###

        ret, img = cap.read()
        #img = cv2.resize(img, (640, 480))
        #print(img.shape)
        
        if img is None:
            break

        ### PREDICTION ###

        # Crop the image

        # For SD
        # Normal case
        img_right = img[50:450, 0:100]
        img_left = cv2.flip(img, 1)
        img_left = img_left[50:450, 0:100]
        img_left = cv2.flip(img_left, 1)

        start = time.time()

        result_right = classify(model, transforms, img_right, classes)
        result_left = classify(model, transforms, img_left, classes)

        end = time.time()
        
        total_time = end - start
        print(f"Prediciton time: {total_time}")

        # Calculate fps
        #fps = 1 / (end - start)

        print_text(img, str(result_right), org=(100, 100))  
        print_text(img, str(result_left), org=(500, 100))
        #print_text(img, f"FPS = {fps:.2f}", org=(150, 1050), color=(0, 0, 255))

        if write:
            out.write(img)

        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.imshow("output", img)
        k = cv2.waitKey(1)
        if k == 27:  # wait for ESC key to exit
            break

    cap.release()
    if write:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # Load the model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
    model.classifier = nn.Linear(1024, NUM_CLASSES)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    #model.half()
    model = model.eval()

    # Run inference
    predict(video_path, out_filename, model, get_transform(), write=False)
