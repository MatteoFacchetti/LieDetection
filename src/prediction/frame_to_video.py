import cv2
import glob
import click
import logging

from utils import file_utils
from utils.model_utils import timer


@click.command()
@click.option("--run_config")
def main(run_config):
    logger.info(f"Run configuration file: {run_config}")
    run_cfg = file_utils.read_yaml(run_config)
    image_size = tuple(run_cfg["modelling"]["image_size"])
    validation_videos = run_cfg["validation_videos"]

    start_time = timer(None)
    for i in validation_videos:
        img_array = []
        for filename in glob.glob(f'../data/validation/0_crop/*_{i}_*.jpg'):
            img = cv2.imread(filename)
            img = cv2.resize(img, image_size[0:2])
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        out = cv2.VideoWriter(f'../data/prediction/video_{i}.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        logger.info(f"Video {i} done...")

    logger.info("Complete!")
    timer(start_time)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    main()
