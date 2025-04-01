import json
from logger import logger


class Suggestion:
    """
    Generates suggestions for enhancing NMT files.
    """

    def __init__(self, suggestion_file="DeepSeek/suggestion.json"):
        self.suggestion_file = suggestion_file

    def update_suggestion(self, filename: str, suggestion: str):
        """
        Update suggestion for a file.
        """
        logger.info(f"Updating suggestion...")

        try:

            try:
                with open(self.suggestion_file, "r") as f:
                    suggestion_data = json.load(f)

            except (FileNotFoundError, json.JSONDecodeError):
                suggestion_data = {}

            # Update suggestion
            suggestion_data[filename] = suggestion

            with open(self.suggestion_file, "w") as f:
                json.dump(suggestion_data, f, indent=4)

            logger.info(f"Suggestion updated successfully")

        except Exception as e:
            logger.exception(f"Error updating suggestion! {str(e)}")
            raise e

    def get_suggestion(self, filename: str):
        """
        Retrieves suggestion for a file.
        """
        logger.info(f"Retrieving suggestion...")

        try:

            with open(self.suggestion_file, "r") as f:
                suggestion_data = json.load(f)

            return suggestion_data.get(filename, "No suggestion found!")

        except Exception as e:
            logger.exception(f"Error retrieving suggestion! {str(e)}")
            raise e
