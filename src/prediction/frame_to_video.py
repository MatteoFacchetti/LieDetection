import cv2
import glob
import click
import logging

from utils import file_utils


@click.command()
@click.option("--run_config")
@click.option("--folder")
@click.option("--input")
def main(run_config, folder, input):
    logger = logging.getLogger(__name__)
    logger.info(f"Run configuration file: {run_config}")
    logger.info(f"Generating video {input} in folder {folder}")

    run_cfg = file_utils.read_yaml(run_config)
    image_size = tuple(run_cfg["modelling"]["image_size"])

    img_array = []
    for filename in glob.glob(f'{folder}/*_{input}_*.jpg'):
        img = cv2.imread(filename)
        img = cv2.resize(img, image_size[0:2])
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    try:
        out = cv2.VideoWriter(f'../data/prediction/video_{input}.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 15, size)
    except UnboundLocalError:
        logger.error("The file may not exist")
        return

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.basicConfig(level=logging.ERROR, format=log_fmt)
    main()