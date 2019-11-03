import yaml


def read_yaml(yaml_file):
    """
    Load a yaml file.

    Parameters
    ----------
    yaml_file : str
        Path to the yaml file to read.

    Returns
    -------
    yml : dict
        The yaml file.
    """
    with open(yaml_file, 'r') as ymlfile:
        yml = yaml.load(ymlfile, yaml.SafeLoader)
    return yml
