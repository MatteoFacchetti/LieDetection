import os
from utils import file_utils

run_cfg = file_utils.read_yaml("../../config/run.yaml")
test_videos = run_cfg["test_videos"]
