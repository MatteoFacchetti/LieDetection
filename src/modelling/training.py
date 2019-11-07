import click
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

import cv2
from imutils import paths
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.layers import Input
from keras.layers.pooling import AveragePooling2D
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import SGD

from utils import file_utils
from utils.model_utils import timer


@click.command()
@click.option("--run_config")
def main(run_config):
    # Initialize
    logger = logging.getLogger(__name__)
    logger.info(f"Run configuration file: {run_config}")

    # Read configuration file
    run_cfg = file_utils.read_yaml(run_config)
    train = run_cfg["train"]
    plot_to_file = run_cfg["evaluation"]["plot_to_file"]
    plot_path = run_cfg["evaluation"]["plot"]
    model_name = run_cfg["modelling"]["model_name"]
    sample_mode = run_cfg["modelling"]["sample_mode"]

    # Grab the list of images and initialize the lists data and images
    logger.info("Loading images...")
    image_paths = list(paths.list_images(train))
    if sample_mode:
        n_frames = 100
        epochs = 5
        image_paths = random.choices(image_paths, k=n_frames)
    else:
        n_frames = 88017
        epochs = run_cfg["modelling"]["epochs"]
    data = []
    labels = []

    # Loop over the image paths
    start_time = timer(None)
    for i in tqdm(range(n_frames)):
        # Extract the class label from the filename
        image_path = image_paths[i]
        label = image_path.split(os.path.sep)[-2]

        # Load the image (reshape e rimappare i colori in base al pretrained model che andiamo a usare)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        # Update data and label lists
        data.append(image)
        labels.append(label)
    logger.info("Images loaded successfully.")
    logger.info(f"Total number of frames loaded: {n_frames}")
    timer(start_time)

    # Convert data and labels to numpy arrays
    logger.info("Processing data...")
    os.environ['KMP_WARNINGS'] = 'off'
    data = np.array(data)
    labels = np.array(labels)

    # Perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    # Train-test split (80%-20%)
    (X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

    # Initialize training data augmentation object
    logger.info("Performing data augmentation...")
    train_aug = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        fill_mode="nearest"
    )

    # Initialize testing data augmentation object
    test_aug = ImageDataGenerator()

    # Center pixel values and intensities (to increase training speed and accuracy)
    # Se si vuole calcolare manualmente l'intensità media dei pixel: X_train.mean(axis=(0, 1, 2))
    mean = np.array([123.68, 116.779, 103.939], dtype="float32")
    train_aug.mean = mean
    test_aug.mean = mean

    # Load ResNet-50 network for fine tuning
    logger.info("Building the model...")
    base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(512, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(1, activation="softmax")(head_model)
    model = Model(inputs=base_model.input, outputs=head_model)

    # Freeze the base model to prevent it from being updated during the training process
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    logger.info("Compiling model...")
    opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / epochs)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Train the head of the network (fine tuning)
    logger.info("Training head of the model...")
    start_time = timer(None)
    head = model.fit_generator(
        train_aug.flow(X_train, y_train, batch_size=32),
        steps_per_epoch=len(X_train) // 32,
        validation_data=test_aug.flow(X_test, y_test),
        validation_steps=len(X_test) // 32,
        epochs=epochs
    )
    timer(start_time)

    # Evaluate the network
    logger.info("Evaluating network...")
    predictions = model.predict(X_test, batch_size=32)
    print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=["Lie"]))

    # Plot train loss and accuracy
    n = epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, n), head.history["loss"], label="train_loss")
    plt.plot(np.arange(0, n), head.history["val_loss"], label="test_loss")
    plt.plot(np.arange(0, n), head.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, n), head.history["val_accuracy"], label="test_acc")
    plt.title("Train and Test Loss and Accuracy")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()
    if plot_to_file:
        plt.savefig(f"{plot_path}/loss_acc.png")

    # Save the model
    logger.info("Saving model...")
    try:
        model.save(f"../models/{model_name}/estimator.model")
    except OSError:
        os.mkdir(f"../models/{model_name}")
        model.save(f"../models/{model_name}/estimator.model")

    # Save the label binarizer
    f = open(f"../models/{model_name}/label_binarizer.pickle", "wb")
    f.write(pickle.dumps(lb))
    f.close()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
