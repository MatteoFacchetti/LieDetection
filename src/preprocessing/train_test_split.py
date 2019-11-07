import click
import logging
import os
import itertools

from utils import file_utils


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
    test_videos = run_cfg["test_videos"]

    # Get name of the frames to move to the test folder
    test_frames = []
    for video in test_videos:
        test_frames.append(list(filter(lambda x: f"_{video}_" in x, os.listdir(f"{train_path}/1_crop"))))
        test_frames.append(list(filter(lambda x: f"_{video}_" in x, os.listdir(f"{train_path}/0_crop"))))
    test_frames = list(itertools.chain(*test_frames))

    # Move test frames to the test folder
    logger.info(f"Moving {len(test_frames)} files...")
    for frame in test_frames:
        try:
            os.rename(f"{train_path}/1_crop/{frame}", f"{test_path}/1_crop/{frame}")
        except FileNotFoundError:
            os.rename(f"{train_path}/0_crop/{frame}", f"{test_path}/0_crop/{frame}")
    logger.info(f"{len(test_frames)} files moved successfully.")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
