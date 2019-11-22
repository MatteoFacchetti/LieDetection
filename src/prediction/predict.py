import click
import logging

from tqdm import tqdm
from keras.models import load_model
from collections import deque
from imutils import paths
import numpy as np
import pickle
import cv2

from utils.model_utils import timer
from utils import file_utils


@click.command()
@click.option("--run_config")
@click.option("--out")
def main(run_config, out):
    # Read configuration file
    run_cfg = file_utils.read_yaml(run_config)
    validation = run_cfg["validation"]
    image_size = tuple(run_cfg["modelling"]["image_size"])

    # Load trained model and label binarizer
    logger.info("Loading model and label binarizer...")
    model = load_model("../models/VGGFaces_16/estimator.model")
    lb = pickle.loads(open("../models/VGGFaces_16/label_binarizer.pickle", "rb").read())

    # Initialize the predictions queue
    q = deque(maxlen=128)

    # Grab the list of images and initialize the lists data and images
    logger.info("Loading validation images...")
    image_paths = list(paths.list_images(validation))[: 600]
    image_paths = [image for i, image in enumerate(image_paths) if i % 3 == 0]
    print(image_paths)

    # Loop over the image paths
    writer = None
    (w, h) = (None, None)
    n_frames = len(image_paths)
    start_time = timer(None)
    for i in tqdm(range(n_frames)):

        # Load the images
        image_path = image_paths[i]
        image = cv2.imread(image_path)

        # If the frame dimensions are empty, grab them
        if w is None or h is None:
            (h, w) = image.shape[:2]
        output = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, image_size[0:2])

        # Make predictions on the frame and update queue
        pred = model.predict(np.expand_dims(image, axis=0))[0]
        q.append(pred)

        # Perform prediction averaging
        results = np.array(q).mean(axis=0)
        r = np.argmax(results)
        label = lb.classes_[r]

        # Draw the activity on the output frame
        text = f"Liar? {label}"
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)

        # Check if the video writer is None
        if writer is None:
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(out, fourcc, 30, True, (w, h))

        # Write the output frame to disk
        writer.write(output)

        # Press q to stop the loop
        # cv2.imshow("Output", output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Clean up and done
    logger.info("Cleaning up...")
    writer.release()
    logger.info("Done")
    timer(start_time)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()