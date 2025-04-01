from CodeLlama.prompt import code_enhance_prompt
from DeepSeek.suggestion import Suggestion
from FAISS.FAISS import FAISSDatabase
from Metadata.metadata import Metadata
from utils.utils import read_py

metadata = Metadata()
faiss_database = FAISSDatabase()
suggestion = Suggestion()


filename = "src/NMT/s1_preprocessing.py"
original_code = read_py(filename=filename)

prompt = code_enhance_prompt(
    filename=filename,
    original_code=original_code,
    suggestion=suggestion.get_suggestion(filename=filename),
    metadata=metadata.get_metadata(),
    example=faiss_database.retrieve_data(query=original_code),
)


print(prompt[0])