import pandas as pd
import click

from utils import file_utils


@click.command()
@click.option("--run_config")
def main(run_config):
    # Read configuration file
    run_cfg = file_utils.read_yaml(run_config)
    test_videos = run_cfg["test_videos"]

    df = pd.read_csv("../data/Annotations.csv")
    y_test = df.loc[test_videos, "truth"]
    print(df.head())
    print(y_test)


if __name__ == "__main__":
    main()
