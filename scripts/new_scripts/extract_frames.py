import cv2
import os, glob

# Opens the Video file
#cap = cv2.VideoCapture('/home/pytholic/Desktop/Projects/uav_mapping/instant-ngp/data/custom_data/videos/ios_videos/IMG_2213.MOV')
#out_dir = '/home/pytholic/Desktop/Projects/uav_mapping/instant-ngp/data/custom_data/images/ios_images/IMG_2213/'
in_dir = '/home/pytholic/Desktop/Projects/datasets/window_detection/data/videos2'
out_dir = '/home/pytholic/Desktop/Projects/datasets/window_detection/data/frames2'

for video in glob.glob(in_dir + '/*.mp4'):
    name = video.split('/')[-1].split('.')[0]
    
    try:
        os.mkdir(os.path.join(out_dir, name))
    except:
        print("An exception occurred")
    
    cap = cv2.VideoCapture(video)
    idx=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        idx += 1
        if (idx % 20) == 0:
            if ret == False:
                break
            #frame = cv2.flip(frame, -1)
            cv2.imwrite(out_dir + '/' + name + '/' + 'frame_' + str(idx) + '.jpg', frame)

cap.release()
cv2.destroyAllWindows()