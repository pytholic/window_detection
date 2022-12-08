import splitfolders

src = '/home/pytholic/Desktop/Projects/datasets/window_detection/data/data_combined'
dst = '/home/pytholic/Desktop/Projects/datasets/window_detection/data/data_final'

splitfolders.ratio(src, output=dst, seed=17, ratio=(.92, .08))