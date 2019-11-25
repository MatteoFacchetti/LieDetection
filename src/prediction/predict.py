import click
import logging

from keras.models import load_model
from collections import deque
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
    image_size = tuple(run_cfg["modelling"]["image_size"])

    # Load trained model and label binarizer
    logger.info("Loading model and label binarizer...")
    model = load_model("../models/VGGFaces_16/estimator.model")
    lb = pickle.loads(open("../models/VGGFaces_16/label_binarizer.pickle", "rb").read())
    lb.classes_ = np.char.replace(lb.classes_, "0_crop", "Lie")
    lb.classes_ = np.char.replace(lb.classes_, "1_crop", "Truth")

    # Initialize the predictions queue
    q = deque(maxlen=128)

    # Loop over the frames in the video
    vs = cv2.VideoCapture("../data/prediction/file.mp4")
    writer = None
    (w, h) = (None, None)
    start_time = timer(None)
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then we have reached the end of the stream
        if not grabbed:
            break

        # If the frame dimensions are empty, grab them
        if w is None or h is None:
            (h, w) = frame.shape[:2]
        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, image_size[0:2])

        # Make predictions on the frame and update queue
        pred = model.predict(np.expand_dims(frame, axis=0))[0]
        q.append(pred)

        # Perform prediction averaging
        results = np.array(q).mean(axis=0)
        r = np.argmax(results)
        label = lb.classes_[r]

        # Draw the activity on the output frame
        text = f"{label}"
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        # Check if the video writer is None
        if writer is None:
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(out, fourcc, 30, (w, h))

        # Write the output frame to disk
        writer.write(output)

    # Clean up and done
    logger.info("Cleaning up...")
    writer.release()
    vs.release()
    logger.info("Done")
    timer(start_time)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()
