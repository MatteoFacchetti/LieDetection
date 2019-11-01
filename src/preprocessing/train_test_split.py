import os
from utils import file_utils

run_cfg = file_utils.read_yaml("../../config/run.yaml")
test_videos = run_cfg["test_videos"]

# Get name of the frames to move to the test folder
test_frames = []
for video in test_videos:
    test_frames.append(list(filter(lambda x: f"_{video}_" in x, os.listdir(f"../../data/0_crop"))))
print(test_frames)

# Move test frames to the test folder
for frame in test_frames:
    try:
        os.rename(f"../../data/0_crop/{frame}", f"../../data/test/0_crop/{frame}")
    except FileNotFoundError:
        os.rename(f"../../data/1_crop/{frame}", f"../../data/test/1_crop/{frame}")
