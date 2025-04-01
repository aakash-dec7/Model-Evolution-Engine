import re
import ollama
from utils.utils import *
from logger import logger
from DeepSeek.prompt import *
from Metadata.metadata import Metadata
from DeepSeek.suggestion import Suggestion


class DeepSeek:
    """
    Generates suggestions for NMT python files.
    """

    def __init__(self, model="deepseek-r1:7b"):
        self.model = model
        self.suggestion = Suggestion()
        self.metadata = Metadata()

    def generate(self, prompt: list):
        """
        Generates suggestion from DeepSeek using the provided prompt.
        """
        response = ollama.chat(model=self.model, messages=prompt)
        return response.message.content

    def generate_suggestion(self, filename: str):
        """
        Reads a python file, generates suggestion and updates them in suggestion.json.
        """
        try:
            # Step1: Read the python file
            code = read_py(filename=filename)

            # Step2: Generate prompt...
            logger.info(f"{filename}: Generating prompt...")
            prompt = generate_suggestion_prompt(
                code=code,
                metadata=self.metadata.get_metadata(),
            )

            # Step3: Generate response from DeepSeek
            logger.info(f"{filename}: Generating response from DeepSeek...")
            response = self.generate(prompt=prompt)

            # Step4: Extract suggestion from response
            logger.info(f"{filename}: Extracting suggestion from response...")

            suggestion = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
            suggestion = suggestion.replace("\n", " ").strip()

            # Step5: Update suggestion
            self.suggestion.update_suggestion(filename=filename, suggestion=suggestion)

        except Exception as e:
            logger.exception(f"{filename}: Error generating suggestion! {str(e)}")
            raise e
