import os
import ast
import json
from logger import logger


class Metadata:
    """
    Extracts metadata from python files in src directory, including class names and fucntion signatures.
    """

    def __init__(self, src_dir="src", metadata_file="Metadata/metadata.json"):
        self.src_dir = src_dir
        self.metadata_file = metadata_file

    def get_function_signature(self, node):
        """
        Extracts fucntion signatures from an AST function node.
        """
        args = [arg.arg for arg in node.args.args]

        return f"{', '.join(args)}"

    def extract_metadata_from_file(self, filename: str):
        """
        Extracts class names and function details from a python file.
        """
        try:
            with open(filename, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename)

        except Exception as e:
            logger.exception(f"Error parsing {filename}: {str(e)}")
            raise e

        metadata = {"classes": [], "functions": []}

        for node in ast.walk(tree):

            if isinstance(node, ast.FunctionDef):
                metadata["functions"].append(
                    {"name": node.name, "signature": self.get_function_signature(node)}
                )

            elif isinstance(node, ast.ClassDef):
                metadata["classes"].append(node.name)

        return metadata

    def extract_metadata_from_dir(self):
        """
        Extract metadata from all python files in the src directory.
        """

        metadata_dict = {}

        if not os.path.exists(self.src_dir):
            logger.warning(f"src directory doesn't exists!")
            return metadata_dict

        for root, _, files in os.walk(self.src_dir):
            for file in files:
                if file.endswith(".py"):

                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.src_dir).replace("\\", "/")

                    metadata = self.extract_metadata_from_file(file_path)

                    if metadata:
                        metadata_dict[relative_path] = metadata

        return metadata_dict

    def update_metadata(self):
        """
        Extracts and saves metadata to metadata.json.
        """
        logger.info("Updating metadata...")

        metadata = self.extract_metadata_from_dir()

        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)

            logger.info("Metadata updated successfully")

        except Exception as e:
            logger.exception(f"Error updating metdata! {str(e)}")
            raise e

    def get_metadata(self):
        """
        Reads and returns metadata from metadata.json.
        """
        logger.info("Getting metadata...")

        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)

        except Exception as e:
            logger.exception(f"Error getting metdata! {str(e)}")
            raise e
