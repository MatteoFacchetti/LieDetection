import cv2
import glob
import click

from imutils import paths

from utils import file_utils


@click.command()
@click.option("--run_config")
@click.option("--out")
def main(run_config, out):
    run_cfg = file_utils.read_yaml(run_config)
    image_size = tuple(run_cfg["modelling"]["image_size"])
    validation = run_cfg["validation"]
    image_paths = list(paths.list_images(validation))[: 600]
    image_paths = [image for i, image in enumerate(image_paths) if i % 3 == 0]

    print(image_paths)
    img_array = []
    for filename in image_paths[: 200]:
        img = cv2.imread(filename)
        img = cv2.resize(img, image_size[0:2])
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
