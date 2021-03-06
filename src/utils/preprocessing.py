import os
import shutil

import cv2


def getframes(folderpath):
    """
    Divide videos in frames given a path that contains a "video.mp4" file.

    Parameters
    ----------
    folderpath : str
        Path to the folder that contains the video.
    """
    vidcap = cv2.VideoCapture(folderpath + "video.mp4")
    success, image = vidcap.read()
    count = 0
    success = True

    # ensure the directory does not exist already
    try:
        shutil.rmtree(folderpath + "frames")
    except:
        pass

    os.makedirs(folderpath + "frames", exist_ok=False)
    while success:
        # save frame as JPEG file
        cv2.imwrite(folderpath + f"frames/frame{count}.jpg", image)
        success, image = vidcap.read()
        count += 1
