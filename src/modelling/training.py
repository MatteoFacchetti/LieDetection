import logging
import os
import numpy as np

import cv2
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

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

# Loop over the image paths
for image_path in image_paths:
    # Extract the class label from the filename
    label = image_path.split(os.path.sep)[-2]

    # Load the image
    # TODO: reshape e rimappare i colori in base al pretrained model che andiamo a usare
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    # Update data and label lists
    data.append(image)
    labels.append(label)
logging.info("Images loaded successfully.")

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# TODO: Train-test split (80%-20%) - Forse non serve, avendo già splittato a monte
(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# TODO: Initialize data augmentation object
