import os
import pandas as pd
from logger import logger
from sklearn.model_selection import train_test_split
from transformers import MarianTokenizer


class Preprocessing:
    """
    Handles text preprocessing, including tokenizer initialization and dataset loading.
    """

    def __init__(self):
        logger.info("Initializing preprocessing...")

        self.tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        self.tokenizer.add_special_tokens({"bos_token": "[BOS]", "eos_token": "[EOS]"})
        self.data = None

    def load_data(self, data_path="src/NMT/artifacts/eng_french.csv", chunksize=5120):
        """
        Loads a dataset from a CSV file efficiently using chunking to reduce memory usage.
        """
        try:
            df_list = []
            for chunk in pd.read_csv(data_path, chunksize=chunksize):
                df = self.preprocess_chunk(chunk)
                df_list.append(df)

            if df_list:
                self.data = pd.concat(df_list, ignore_index=True)
                logger.info("Dataset loaded successfully with chunking")
                return self.data
            else:
                raise ValueError("No data found in the file.")

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise e

    def preprocess_chunk(self, chunk):
        """
        Preprocesses a single chunk of data.
        """
        return (
            chunk  # Modify this function for additional preprocessing steps if needed
        )

    def tokenize_data(self, data):
        """
        Tokenizes English and French text data using the MarianTokenizer.
        """
        try:
            input_data = data["English"]
            target_data = data["French"].apply(
                lambda x: f"{self.tokenizer.bos_token} {x} {self.tokenizer.eos_token}"
            )

            input_seq = self.tokenizer(
                input_data.tolist(),
                padding="max_length",
                max_length=25,
                truncation=True,
                return_tensors="pt",
            ).input_ids

            target_seq = self.tokenizer(
                target_data.tolist(),
                padding="max_length",
                max_length=25,
                truncation=True,
                return_tensors="pt",
            ).input_ids

            logger.info("Dataset tokenized successfully")
            return input_seq, target_seq

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise e

    def train_test_split(self, input_seq, target_seq):
        """
        Splits tokenized input and target sequences into training and testing sets.
        """
        try:
            return train_test_split(
                input_seq, target_seq, test_size=0.2, random_state=21
            )
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise e

    def save_data(self, dir, input_data, target_data):
        """
        Saves tokenized input and target data as CSV files.
        """
        try:
            data_dir = f"src/NMT/artifacts/{dir}"
            os.makedirs(data_dir, exist_ok=True)

            pd.DataFrame(input_data).to_csv(
                f"{data_dir}/input_data.csv", index=False, header=False
            )
            pd.DataFrame(target_data).to_csv(
                f"{data_dir}/target_data.csv", index=False, header=False
            )

            logger.info("Dataset saved successfully")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise e

    def save_tokenizer(self):
        """
        Saves the tokenizer in a specified directory.
        """
        try:
            tokenizer_dir = "src/NMT/artifacts/tokenizer"
            os.makedirs(tokenizer_dir, exist_ok=True)
            self.tokenizer.save_pretrained(tokenizer_dir)
            logger.info("Tokenizer saved successfully")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise e

    def run(self):
        """
        Executes the preprocessing pipeline.
        """
        try:
            self.data = self.load_data()
            tokenized_input_data, tokenized_target_data = self.tokenize_data(self.data)
            train_input_data, test_input_data, train_target_data, test_target_data = (
                self.train_test_split(tokenized_input_data, tokenized_target_data)
            )
            self.save_data("train", train_input_data, train_target_data)
            self.save_data("test", test_input_data, test_target_data)
            self.save_tokenizer()
            logger.info("Preprocessing pipeline completed successfully")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise e


if __name__ == "__main__":
    try:
        preprocessing = Preprocessing()
        preprocessing.run()
    except Exception as e:
        raise RuntimeError("Preprocessing pipeline failed!") from e
