import os
import shutil
import pandas as pd

from utils.file_utils import getframes

annotations = pd.read_csv("../../data/Annotations.csv")

for i, folderpath in enumerate(annotations.video):
    getframes(folderpath.replace("video.mp4", ""))
    if (i + 1) % 10 == 0:
        print(f"done with the {i + 1}th folderpath")

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
    for i, folderpath in zip(temp.index, temp.video):
        # iterate through each frame for the given path and move it to the folder labeled 0 or 1
        for frame in os.listdir(folderpath.replace("video.mp4", "") + "frames"):
            shutil.copy(folderpath.replace("video.mp4", "") + "frames/" + frame,
                        f"./{label}/{i}_{frame}")
