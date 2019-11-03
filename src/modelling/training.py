import logging
import os
import matplotlib.pyplot as plt

import cv2
from imutils import paths

from utils import file_utils

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Read configuration files
run_cfg = file_utils.read_yaml("../../config/run.yaml")
train = run_cfg["train"]

# Grab the list of images and initialize the lists data and images
logging.info("Loading images...")
image_paths = list(paths.list_images(train))
data = []
labels = []

# TODO: Loop over the image paths
for image_path in image_paths:
    # Extract the class label from the filename
    label = image_path.split(os.path.sep)[-2]

    # Load the image
    # TODO: reshape e rimappare i colori in base al pretrained model che andiamo a usare
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    b, g, r = cv2.split(image)
    frame_rgb = cv2.merge((r, g, b))
    plt.imshow(frame_rgb)
    plt.title('Matplotlib')  # Give this plot a title,
    # so I know it's from matplotlib and not cv2
    plt.show()
    break

    # Update data and labels
    data.append(image)
    labels.append(label)
