from CodeLlama.model import CodeLlama
from DeepSeek.model import DeepSeek
from Metadata.metadata import Metadata


def process_files(file_list):
    """
    Updates metadata, generates suggestions, and enhances code for given files.
    """
    Metadata().update_metadata()

    deepseek = DeepSeek()
    codellama = CodeLlama()

    for file in file_list:
        deepseek.generate_suggestion(filename=file)
        codellama.enhance(filename=file)


if __name__ == "__main__":
    file_list = [
        "src/NMT/model.py",
        "src/NMT/s1_preprocessing.py",
        "src/NMT/s2_training.py",
        "src/NMT/s3_evaluation.py",
    ]
    process_files(file_list)
