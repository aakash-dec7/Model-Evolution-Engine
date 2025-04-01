import os
import json
import textwrap
from logger import logger


def read_py(filename: str):
    """
    Reads code from a python file.
    """
    logger.info(f"{filename}: Reading code...")

    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()

    except Exception as e:
        logger.exception(f"{filename}: {str(e)}")
        raise e


def write_py(filename: str, code: str):
    """
    Writes code to a python file.
    """
    logger.info(f"{filename}: Writing code...")

    try:
        formatted_code = textwrap.dedent(code).strip()

        with open(filename, "w", encoding="utf-8") as f:
            return f.write(formatted_code)

    except Exception as e:
        logger.exception(f"{filename}: {str(e)}")
        raise e


def create_py(filename: str):
    """
    Creates a python file.
    """
    logger.info(f"{filename}: Creating python file...")

    try:
        open(filename, "w").close()

    except Exception as e:
        logger.exception(f"{filename}: {str(e)}")
        raise e


def update_metrics(filename: str, key: str, value: str, metric_file="metrics.json"):
    """
    Updates metrics.json file with a nested structure: {filename: {key: value}}.
    """
    data = {}

    # Load existing data if the file exists
    if os.path.exists(metric_file):
        with open(metric_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}  # Reset if file is corrupted

    # Ensure the filename key exists
    if filename not in data:
        data[filename] = {}

    # Update the nested dictionary
    data[filename][key] = value

    # Save back to JSON
    with open(metric_file, "w") as f:
        json.dump(data, f, indent=4)
