import click
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
import warnings
warnings.filterwarnings('ignore')

import cv2
from imutils import paths
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras_vggface.vggface import VGGFace
from keras.layers import Input
from keras.layers.pooling import AveragePooling2D
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import CSVLogger, EarlyStopping

from utils import file_utils
from utils.model_utils import timer


@click.command()
@click.option("--run_config")
def main(run_config):
    # Initialize
    logger.info(f"Run configuration file: {run_config}")

    # Read configuration file
    run_cfg = file_utils.read_yaml(run_config)
    train = run_cfg["train"]
    test = run_cfg["test"]
    model_name = run_cfg["modelling"]["model_name"]
    sample_mode = run_cfg["modelling"]["sample_mode"]
    data_augmentation = run_cfg["modelling"]["data_augmentation"]
    image_size = tuple(run_cfg["modelling"]["image_size"])
    vgg_model = run_cfg["modelling"]["vgg_model"]

    # Load train and test images
    X_train, y_train, epochs = initialize(train, run_cfg, image_size, sample_mode)
    X_test, y_test, epochs = initialize(test, run_cfg, image_size, sample_mode)

    # Perform one-hot encoding on the labels
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    lb = LabelBinarizer()
    y_test = lb.fit_transform(y_test)

    # Initialize training data augmentation object
    if data_augmentation:
        logger.info("Performing data augmentation...")
        train_aug = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            fill_mode="nearest"
        )
    else:
        logger.info("Data augmentation not performed")
        train_aug = ImageDataGenerator(rescale=1./255)
    test_aug = ImageDataGenerator(rescale=1./255)

    # Load VGGFace network for fine tuning
    logger.info("Building the model...")
    base_model = VGGFace(model=vgg_model, weights="vggface", include_top=False, input_tensor=Input(shape=image_size[0:3]))
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=image_size[3:5])(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(512, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(1, activation="sigmoid")(head_model)
    model = Model(inputs=base_model.input, outputs=head_model)

    # Freeze the base model to prevent it from being updated during the training process
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    logger.info("Compiling model...")
    opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / epochs)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Callbacks: early stopping and CSVLogger
    earlystop = EarlyStopping(
        monitor='val_acc',
        min_delta=0.001,
        patience=10,
        verbose=1,
        mode='auto'
    )
    if os.path.exists(f"../models/{model_name}"):
        csv_logger = CSVLogger(f"../models/{model_name}/CSVLogger.log")
    else:
        os.mkdir(f"../models/{model_name}")
        csv_logger = CSVLogger(f"../models/{model_name}/CSVLogger.log")

    # Train the head of the network (fine tuning)
    logger.info("Training head of the model...")
    start_time = timer(None)
    head = model.fit_generator(
        train_aug.flow(X_train, y_train, batch_size=20),
        steps_per_epoch=len(X_train) // 20,
        validation_data=test_aug.flow(X_test, y_test),
        validation_steps=len(X_test) // 20,
        epochs=epochs,
        callbacks=[earlystop, csv_logger]
    )
    timer(start_time)

    # Evaluate the network
    logger.info("Evaluating network...")
    predictions = model.predict(X_test, batch_size=20)
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0
    print(classification_report(y_test, predictions))

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

    # Save the model
    if not sample_mode:
        logger.info("Saving model...")
        model.save(f"../models/{model_name}/estimator.model")

        # Save the label binarizer
        f = open(f"../models/{model_name}/label_binarizer.pickle", "wb")
        f.write(pickle.dumps(lb))
        f.close()

        # Save plot
        plt.savefig(f"../models/{model_name}/loss_acc.png")
        logger.info("Model saved successfully.")


def initialize(group, run_cfg, image_size, sample_mode):
    # Grab the list of images and initialize the lists data and images
    logger.info("Loading images...")
    image_paths = list(paths.list_images(group))
    image_paths = [image for i, image in enumerate(image_paths) if i % 3 == 0]
    if sample_mode:
        n_frames = 100
        epochs = 5
        image_paths = random.choices(image_paths, k=n_frames)
    else:
        n_frames = len(image_paths)
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
        image = cv2.resize(image, image_size[0:2])

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

    return data, labels, epochs


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()
