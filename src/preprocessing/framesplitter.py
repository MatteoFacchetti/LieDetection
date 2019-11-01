import cv2
import os
import shutil
import pandas as pd

annotations = pd.read_csv("Annotations.csv")


# script divides videos in frames given a path that contains a "video.mp4" file
def getframes(folderpath):
    vidcap = cv2.VideoCapture(folderpath + "video.mp4")
    success,image = vidcap.read()
    count = 0
    success = True
    
    # ensure the directory does not exist already
    try:    
        shutil.rmtree(folderpath + "frames") 
    except:
        pass
    
    os.makedirs(folderpath + "frames", exist_ok = False)
    while success:
        # save frame as JPEG file
        cv2.imwrite(folderpath + f"frames/frame{count}.jpg", image)
        success,image = vidcap.read()
        count += 1


for i, folderpath in enumerate(annotations.video):
    getframes(folderpath.replace("video.mp4", ""))
    if (i+1)%10 == 0:
        print(f"done with the {i+1}th folderpath")


# then distribute the frames in two folders
for label in [0, 1]:
    try:    
        shutil.rmtree(f"{label}")
    except:
        pass
    os.mkdir(f"{label}")

# iterate through the two possible dummy values
for label in [0, 1]:
    temp = annotations[annotations.truth == label]
    # iterate through each path where the dummy variable assumes the value under consideration
    for i, folderpath in enumerate(temp.video):
        # iterate through each frame for the given path and move it to the folder labeled 0 or 1
        for frame in os.listdir(folderpath.replace("video.mp4", "") + "frames"):
            shutil.copy(folderpath.replace("video.mp4", "") + "frames/" + frame,
                     f"./{label}/{i}_{frame}")
