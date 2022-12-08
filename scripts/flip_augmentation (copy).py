import os, glob
import numpy as np

in_dir = '/home/pytholic/Desktop/Projects/window_detection/data_cropped/open'

img_list = glob.glob(in_dir + '/right_*.jpg')
for i in range(196):
    idx = np.random.randint(0, len(img_list))
    img = str(img_list[idx])
    os.remove(img)    