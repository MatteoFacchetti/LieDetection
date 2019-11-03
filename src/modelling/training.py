import logging
import os

import cv2
from imutils import paths

from utils import file_utils

# Logging configuration
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

run_cfg = file_utils.read_yaml("../../config/run.yaml")
logging.info(run_cfg)
train = run_cfg["train"]

# Grab the list of images and initialize the lists data and images
logging.info("Loading images...")
image_paths = list(paths.list_images(train))
data = []
labels = []

# Loop over the image paths
for image_path in image_paths:
    # Extract the class label from the filename
    label = image_path.split(os.path.sep)[-2]

    # Load the image
    # TODO: controllare la dimensione delle immagini e controllare che siano RGB
    image = cv2.imread(image_path)

    # Update data and labels
    data.append(image)
    labels.append(label)
