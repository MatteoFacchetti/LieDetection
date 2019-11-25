import pandas as pd
import click
import logging
import pickle
import glob
import cv2
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['KMP_WARNINGS'] = 'off'

from keras.models import load_model

from utils import file_utils
from utils.model_utils import timer


@click.command()
@click.option("--run_config")
def main(run_config):
    # Read configuration file
    logger = logging.getLogger(__name__)
    logger.info(f"Run configuration file: {run_config}")
    run_cfg = file_utils.read_yaml(run_config)
    test_videos = run_cfg["test_videos"]
    validation_videos = run_cfg["validation_videos"]
    image_size = tuple(run_cfg["modelling"]["image_size"])

    # Load true labels
    logger.info("Loading true labels")
    df = pd.read_csv("../data/Annotations.csv")
    y_test = df.loc[test_videos, "truth"]
    y_vali = df.loc[validation_videos, "truth"]

    # Load trained model and label binarizer
    logger.info("Loading trained model and label binarizer")
    logger.info("Loading model and label binarizer...")
    model = load_model("../models/VGGFaces_16/estimator.model")
    lb = pickle.loads(open("../models/VGGFaces_16/label_binarizer.pickle", "rb").read())

    # Load test frames
    logger.info("Making predictions")
    start_time = None
    preds = dict()

    # 0_crop predictions
    for i in test_videos:
        predictions = []
        for filename in glob.glob(f'../data/test/0_crop/*_{i}_*.jpg'):
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size[0:2])

            # Make predictions on the frame and update queue
            pred = model.predict(np.expand_dims(img, axis=0))[0]
            predictions.append(pred)

        if predictions:
            # Perform prediction averaging
            results = np.array(predictions).mean(axis=0)
            r = np.argmax(results)
            label = lb.classes_[r]

            preds[f"{i}"] = label
            logger.info(f"Prediction for video {i} done")
        timer(start_time)

    # 1_crop predictions
    for i in test_videos:
        predictions = []
        for filename in glob.glob(f'../data/test/1_crop/*_{i}_*.jpg'):
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size[0:2])

            # Make predictions on the frame and update queue
            pred = model.predict(np.expand_dims(img, axis=0))[0]
            predictions.append(pred)

        if predictions:
            # Perform prediction averaging
            results = np.array(predictions).mean(axis=0)
            r = np.argmax(results)
            label = lb.classes_[r]

            preds[f"{i}"] = label
            logger.info(f"Prediction for video {i} done")
        timer(start_time)

    preds = pd.DataFrame(preds).transpose()
    preds.to_csv("predictions.csv")


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
