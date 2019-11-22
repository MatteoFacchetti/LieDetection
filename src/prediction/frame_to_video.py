import cv2
import glob
import click

from utils import file_utils


@click.command()
@click.option("--run_config")
@click.option("--out")
def main(run_config, out):
    run_cfg = file_utils.read_yaml(run_config)
    image_size = tuple(run_cfg["modelling"]["image_size"])

    img_array = []
    for filename in glob.glob(f'../data/validation/0_crop/*_{120}_*.jpg'):
        img = cv2.imread(filename)
        img = cv2.resize(img, image_size[0:2])
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('../data/prediction/file.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == "__main__":
    main()
