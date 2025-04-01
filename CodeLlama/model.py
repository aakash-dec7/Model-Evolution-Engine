import re
import ollama
import textwrap
from utils.utils import *
from logger import logger
from CodeLlama.prompt import *
from FAISS.FAISS import FAISSDatabase
from Metadata.metadata import Metadata
from DeepSeek.suggestion import Suggestion
from CodeLlama.evaluate import EvaluateAICode


class CodeLlama:
    """
    Enhances NMT python files based on suggestions provided by DeepSeek.
    """

    def __init__(self, model="codellama:7b"):
        self.model = model
        self.metadata = Metadata()
        self.faiss_database = FAISSDatabase()
        self.evaluate_ai_code = EvaluateAICode()
        self.suggestion = Suggestion()

    def generate(self, prompt: list):
        """
        Generates a response from CodeLlama using the provided prompt.
        """
        response = ollama.chat(model=self.model, messages=prompt)
        return response.message.content

    def enhance(self, filename: str):
        """
        Enhances the NMT python files based on DeepSeek's suggestions.
        """
        attempt = 1
        prompt = None

        try:
            while attempt <= 5:
                logger.info(f"{filename}: Attempt: {attempt}")

                # Step1: Read the python file
                original_code = read_py(filename=filename)
                if not original_code:
                    logger.error(f"{filename}: Failed to read file!")
                    return False

                # Step2: Generate initial prompt
                if prompt is None:
                    logger.info(f"{filename}: Generating initial prompt...")
                    prompt = code_enhance_prompt(
                        filename=filename,
                        original_code=original_code,
                        suggestion=self.suggestion.get_suggestion(filename=filename),
                        metadata=self.metadata.get_metadata(),
                        example=self.faiss_database.retrieve_data(query=original_code),
                    )

                # Step3: Generate response from CodeLlama
                logger.info(f"{filename}: Generating response from CodeLlama...")
                response = self.generate(prompt=prompt)

                # Step4: Extract Ai code from response
                logger.info(f"{filename}: Extracting Ai code from response...")

                matches = re.findall(r"```python\n([\s\S]*?)\n```", response, re.DOTALL)
                ai_code = (
                    textwrap.dedent(matches[0]).strip() if matches else response.strip()
                )

                if not ai_code:
                    logger.error(f"{filename}: Ai code is empty! Retrying...")
                    attempt += 1
                    continue

                # Step5: Evaluate Ai code
                eval_status, reward, error = self.evaluate_ai_code.evaluate(
                    filename=filename,
                    original_code=original_code,
                    ai_code=ai_code,
                )

                if eval_status:
                    write_py(filename=filename, code=ai_code)

                    # Update metadata
                    self.metadata.update_metadata()

                    # Update FAISS database if reward meets threshold
                    if reward >= 25:
                        suggestions = " ".join(
                            self.suggestion.get_suggestion(filename=filename),
                        )
                        data = f"{suggestions} {ai_code}"
                        self.faiss_database.update_database(data=data)

                    logger.info(f"{filename}: Enhancement successful...")
                    return True

                logger.error(f"{filename}: Enhancement failed! Retrying...")

                # Step6: Generate new prompt with error context
                logger.info(f"{filename}: Generating prompt with error context...")

                prompt = code_enhance_prompt_with_potential_error(
                    filename=filename,
                    original_code=original_code,
                    suggestion=self.suggestion.get_suggestion(filename=filename),
                    metadata=self.metadata.get_metadata(),
                    example=self.faiss_database.retrieve_data(query=original_code),
                    potential_error=error,
                )

                attempt += 1

            logger.error(f"{filename}: Maximum attempt(5) reached. Enhancement failed!")

            return False

        except Exception as e:
            logger.exception(f"{filename}: Error enhancing! {str(e)}")
            raise e
