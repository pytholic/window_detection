import os, glob
import cv2

in_dir = '/home/pytholic/Desktop/Projects/window_detection/data'
out_dir = '/home/pytholic/Desktop/Projects/window_detection/data_cropped'
CASE = 'close'  # 'open' or 'close'
WINDOW = 'right'  # 'right' or 'left'

if CASE == 'open':
    dst = os.path.join(out_dir, CASE)
elif CASE == 'close':
    dst = os.path.join(out_dir, CASE)

for image in glob.glob(os.path.join(in_dir, CASE) + '/*.jpg'):
    """
    Image size: (1080, 1920)
    Right crop: img[300:900, 1450:1700]
    """
    name = image.split('/')[-1]
    img = cv2.imread(image)
    
    if WINDOW == 'right':
        img = img[300:900, 1450:1700]
        cv2.imwrite(os.path.join(dst, 'right_' + name), img)
    
    elif WINDOW == 'left':
        img = cv2.flip(img, 1)
        img = img[300:900, 1450:1700]
        img = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(dst, 'left_' + name), img)

    # cv2.imshow('Image', img)    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
