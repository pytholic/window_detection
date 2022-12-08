import os, glob
import numpy as np
import shutil

# Input and output paths
data_dir = '/home/pytholic/Desktop/Projects/icms_data/data_new/images'
out_dir = '/home/pytholic/Desktop/Projects/window_detection/data'

# Index list of videos
# Open case
open_list_1 = list(range(1, 37))
open_list_2 = list(range(38, 51))
open_list_3 = list((51, 53, 54))
open_list_4 = list((68, 70))
open_list = sorted([y for x in [open_list_1, open_list_2, open_list_3, open_list_4] for y in x])

# Close case
close_list = [37, 51, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70]

# Choose `open` or `close`
CASE = 'open'
num_images = 6  # 6 for open, 20 for close

if CASE == 'open':
    out_path = os.path.join(out_dir, CASE)
    folder_list = open_list
elif CASE == 'close':
    out_path = os.path.join(out_dir, CASE)
    folder_list = close_list
    
for folder in folder_list:
    try:
        in_path = os.path.join(data_dir, '0' + str(folder))
        img_list = glob.glob(in_path + '/*.jpg')
        for i in range(num_images):
            idx = np.random.randint(0, len(img_list))
            src = str(img_list[idx])
            dst = os.path.join(out_path, str(img_list[idx].split('/')[-1]))
            shutil.copy(src, dst)
    except:
        pass