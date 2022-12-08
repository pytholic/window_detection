import os, glob
import cv2

in_dir = '/home/pytholic/Desktop/Projects/datasets/window_detection/data/frames/0015'

for image in glob.glob(in_dir + '/*.jpg'):
    """
    Image size: (1080, 1920)
    Right crop: img[300:900, 1450:1700]
    """
    name = image.split('/')[-1]
    img = cv2.imread(image)
    
    # for tinted
    #img = img[232:285, 580:690]  # 037
    #img = img[259:293, 1316:1377]  # 037
    #img = img[163:243, 1292:1381]  # 051
    #img = img[162:197, 595:631]  # 051
    
    # for not tinted
    # For HD
    img_right = img[300:900, 1450:1700]
    img_left = cv2.flip(img, 1)
    img_left = img_left[300:900, 1450:1700]
    img_left = cv2.flip(img_left, 1)
    img_right_back = img[222:265, 1200:1250]

    cv2.imshow('Image', img)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    break
