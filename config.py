import yaml


def load_config(yaml_file_path):
    """
    Loads and parses a YAML configuration file.

    Args:
        yaml_file_path (str): The path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.
    """
    try:
        with open(yaml_file_path, "r",encoding='utf-8') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        print(f"The file {yaml_file_path} was not found.")
        return {}
    except yaml.YAMLError as exc:
        print(f"Error in configuration file: {exc}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}


def check_config(config_dict):
    pass


# 使用示例
if __name__ == "__main__":
    config_file_path = "config.yaml"
    config = load_config(config_file_path)
    print(config)
