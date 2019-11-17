import click
import logging
import os
import itertools

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
    train_path = run_cfg["train"]
    test_path = run_cfg["test"]
    validation_path = run_cfg["validation"]
    test_videos = run_cfg["test_videos"]
    validation_videos = run_cfg["validation_videos"]
    start_time = timer(None)

    # Get name of the frames to move to the test and validation folder
    test_frames = []
    for video in test_videos:
        test_frames.append(list(filter(lambda x: f"_{video}_" in x, os.listdir(f"{train_path}/1_crop"))))
        test_frames.append(list(filter(lambda x: f"_{video}_" in x, os.listdir(f"{train_path}/0_crop"))))
    test_frames = list(itertools.chain(*test_frames))

    validation_frames = []
    for video in validation_videos:
        validation_frames.append(list(filter(lambda x: f"_{video}_" in x, os.listdir(f"{train_path}/1_crop"))))
        validation_frames.append(list(filter(lambda x: f"_{video}_" in x, os.listdir(f"{train_path}/0_crop"))))
    validation_frames = list(itertools.chain(*validation_frames))

    # Move test and validation frames to test and validation folders
    logger.info(f"Moving {len(test_frames)} test files...")
    for frame in test_frames:
        try:
            os.rename(f"{train_path}/1_crop/{frame}", f"{test_path}/1_crop/{frame}")
        except FileNotFoundError:
            os.rename(f"{train_path}/0_crop/{frame}", f"{test_path}/0_crop/{frame}")
    logger.info(f"{len(test_frames)} files moved successfully.")

    logger.info(f"Moving {len(validation_frames)} validation files...")
    for frame in validation_frames:
        try:
            os.rename(f"{train_path}/1_crop/{frame}", f"{validation_path}/1_crop/{frame}")
        except FileNotFoundError:
            os.rename(f"{train_path}/0_crop/{frame}", f"{validation_path}/0_crop/{frame}")
    logger.info(f"{len(validation_frames)} files moved successfully.")
    timer(start_time)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
