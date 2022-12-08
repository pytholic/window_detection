import os, glob
import cv2

in_dir = '/home/pytholic/Desktop/Projects/window_detection/data_cropped'
out_dir = '/home/pytholic/Desktop/Projects/window_detection/data_cropped/temp'
CASE = 'close'  # 'open' or 'close'

for image in glob.glob(os.path.join(in_dir, CASE) + '/*.jpg'):

    name = str(image.split('/')[-1])
    img = cv2.imread(image)
    
    if 'left' in name:
        img = cv2.flip(img, 1)
        new_name = name.replace('left_', 'right_aug_')
        cv2.imwrite(os.path.join(out_dir, new_name), img)
    
    elif 'right' in name:
        img = cv2.flip(img, 1)
        new_name = name.replace('right_', 'left_aug_')
        cv2.imwrite(os.path.join(out_dir, new_name), img)