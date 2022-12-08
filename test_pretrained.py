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
video_source = '/dev/video0'
NUM_CLASSES = 2
classes = ["open", "close"]

four_k = (3840, 2160)
FULL_HD = (1920,1080)
SD = (640, 480)

#probe = ffmpeg.probe(video_path)
probe = ffmpeg.probe(video_source)
video_info = next(x for x in probe["streams"] if x["codec_type"] == "video")
WIDTH = int(video_info["width"])
HEIGHT = int(video_info["height"])


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

### ffmpeg utilities ###

# Read functions
def ffmpeg_reading_process(filename):
    args = (
        #ffmpeg.input(filename)
        ffmpeg.input(filename=video_source)
        .output("pipe:", crf=0, preset="ultrafast", format="rawvideo", pix_fmt="rgb24")
        .compile()
    )
    return sp.Popen(args, stdout=sp.PIPE)


def read_frame(process, width, height):
    frame_size = width * height * 3
    in_bytes = process.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
    return frame


# Write functions
def ffmpeg_writing_process(filename, width, height):
    args = (
        ffmpeg.input(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="{}x{}".format(width, height)
        )
        .filter("fps", fps=30, round="up")
        # .setpts('1.2*PTS')
        .output(
            filename, crf=0, preset="ultrafast", movflags="faststart", pix_fmt="rgb24"
        )
        .overwrite_output()
        .compile()
    )
    return sp.Popen(args, stdin=sp.PIPE)


def write_frame(process, frame):
    process.stdin.write(frame.astype(np.uint8).tobytes())


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

def predict_opencv(video_path, out_filename, model, transforms, write=False):

    if write:

        ext = str(out_filename.split("/")[-1].split(".")[-1])
        if ext == "mp4":
            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        else:
            fourcc = cv2.VideoWriter_fourcc("x", "v", "i", "d")

        out = cv2.VideoWriter(out_filename, fourcc, 30.0, (WIDTH, HEIGHT))
    
    #cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(2)
    res = SD
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    print(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    while True:

        ### INPUT ###

        ret, img = cap.read()
        img = cv2.resize(img, (640, 480))
        #print(img.shape)
        #img = cv2.resize(img, (1920, 1080))
        
        if img is None:
            break

        ### PREDICTION ###

        # Crop the image

        # For 4k
        # img_right = img[900:1500, 3150:3450]
        # img_left = cv2.flip(img, 1)
        # img_left = img_left[900:1500, 2950:3250]
        # img_left = cv2.flip(img_left, 1)
        # img_right = img[700:1500, 2900:3500]
        # img_left = cv2.flip(img, 1)
        # img_left = img_left[700:1500, 2900:3500]
        # img_left = cv2.flip(img_left, 1)

        # For Full HD
        # img_right = img[300:900, 1450:1700]
        # img_left = cv2.flip(img, 1)
        # img_left = img_left[300:900, 1450:1700]
        # img_left = cv2.flip(img_left, 1)

        # For SD
        # Normal case
        # img_right = img[50:450, 0:100]
        # img_left = cv2.flip(img, 1)
        # img_left = img_left[50:450, 0:100]
        # img_left = cv2.flip(img_left, 1)

        # Resized case from full HD
        img_right = img[50:450, 50:150]
        img_left = cv2.flip(img, 1)
        img_left = img_left[50:450, 40:140]
        img_left = cv2.flip(img_left, 1)

        start = time.time()

        result_right = classify(model, transforms, img_right, classes)
        result_left = classify(model, transforms, img_left, classes)

        end = time.time()

        # Calculate fps
        fps = 1 / (end - start)

        print_text(img, str(result_right), org=(100, 100))  
        print_text(img, str(result_left), org=(300, 100))
        # print_text(img, str(result_right), org=(1600, 400))
        # print_text(img, str(result_left), org=(300, 400))
        print_text(img, f"FPS = {fps:.2f}", org=(150, 1050), color=(0, 0, 255))

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

# Predict with ffmpeg
def predict_ffmpeg(video_path, out_filename, model, transforms, write=False):

    process_read = ffmpeg_reading_process(video_path)
    if write:
        process_write = ffmpeg_writing_process(out_filename, WIDTH, HEIGHT)

    while True:

        ### INPUT ###

        frame = read_frame(process_read, WIDTH, HEIGHT)
        frame = cv2.resize(frame, (640, 480))

        if frame is None:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ### PREDICTION ###

        # Crop the image

        # For 4k
        # img_right = frame[900:1500, 3150:3450]
        # img_left = cv2.flip(frame, 1)
        # img_left = img_left[900:1500, 2950:3250]
        # img_left = cv2.flip(img_left, 1)

        # For Full HD
        # img_right = img[300:900, 1450:1700]
        # img_left = cv2.flip(img, 1)
        # img_left = img_left[300:900, 1450:1700]
        # img_left = cv2.flip(img_left, 1)

        # For SD
        img_right = img[150:350, 0:70]
        img_left = cv2.flip(img, 1)
        img_left = img_left[150:350, 0:70]
        img_left = cv2.flip(img_left, 1)

        start = time.time()

        result_right = classify(model, get_transform(), img_right, classes)
        result_left = classify(model, get_transform(), img_left, classes)
        #result_left_back = classify(model, get_transform(), img_left_back, classes)
        #result_right_back = classify(model, get_transform(), img_right_back, classes)
        
        end = time.time()

        # Calculate fps
        fps = 1 / (end - start)

        print_text(img, str(result_right), org=(100, 100))  
        print_text(img, str(result_left), org=(500, 100))
        # print_text(img, str(result_right), org=(1600, 400))  
        # print_text(img, str(result_left), org=(300, 400))
        # print_text(img, str(result_right), org=(3200, 800))  
        # print_text(img, str(result_left), org=(600, 800))
        #print_text(img, str(result_left_back), org=(600, 200))
        #print_text(img, str(result_right_back), org=(1300, 200))
        #print_text(img, f"FPS = {fps:.2f}", org=(150, 1050), color=(0, 0, 255))

        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.imshow("output", img)
        k = cv2.waitKey(1)
        if k == 27:  # wait for ESC key to exit
            break

        if write:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            write_frame(process_write, img)

    cv2.destroyAllWindows()
    process_read.terminate()
    if write:
        process_write.terminate()


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
    predict_opencv(video_path, out_filename, model, get_transform(), write=False)
    #predict_ffmpeg(video_path, out_filename, model, get_transform(), write=False)
