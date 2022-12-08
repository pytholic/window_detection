import os, glob
import cv2

in_dir = '/home/pytholic/Desktop/Projects/datasets/window_detection/data/frames2'
out_dir = '/home/pytholic/Desktop/Projects/datasets/window_detection/data/data_cropped'

for idx, folder in enumerate(os.listdir(in_dir)):
    idx+=26
    path = os.path.join(in_dir, folder)
    for image in glob.glob(path + '/*.jpg'):
        name = image.split('/')[-1]
        img = cv2.imread(image)
        img_right = img[300:900, 1450:1700]
        img_left = cv2.flip(img, 1)
        img_left = img_left[300:900, 1450:1700]
        img_left = cv2.flip(img_left, 1)
        # img_right = img[900:1500, 3150:3450]
        # img_left = cv2.flip(img, 1)
        # img_left = img_left[900:1500, 2950:3250]
        # img_left = cv2.flip(img_left, 1)
        cv2.imwrite(os.path.join(out_dir, str(idx) + '_right_' + name), img_right)
        cv2.imwrite(os.path.join(out_dir, str(idx) + '_left_' + name), img_left)
