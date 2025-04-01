import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from logger import logger


class FAISSDatabase:
    """
    Creates a FAISS Vector-Database to store data locally.
    """

    def __init__(self, model="all-MiniLM-L6-v2", index_file="FAISS/FAISS.index"):
        self.d = 384
        self.index = faiss.IndexFlatL2(self.d)
        self.model = SentenceTransformer(model)
        self.index_file = index_file
        self.data = []

    def update_database(self, data: str):
        """
        Encodes the input data, adds it to FAISS index and saves the index to FAISS.index.
        """
        logger.info("Updating FAISS database...")

        try:
            embedded_data = np.array(self.model.encode([data]), dtype=np.float32)

            # Add embedded_data to FAISS index
            self.index.add(embedded_data)

            # Save index to FAISS.index
            faiss.write_index(self.index, self.index_file)

        except Exception as e:
            logger.exception(f"Error updating FAISS database: {str(e)}")
            raise e

    def retrieve_data(self, query: str, k=1):
        """
        Retrieves the top-k most similar data from FAISS.index based on the query.
        """
        logger.info("Retrieving data from FAISS database...")

        try:
            embedded_query = np.array(self.model.encode([query]), dtype=np.float32)

            # Search in FAISS.index
            distances, indices = self.index.search(embedded_query, k=k)

            # Retrieve data
            retrieved_data = [self.data[i] for i in indices[0] if i != -1]

            return retrieved_data

        except Exception as e:
            logger.exception(f"Error retrieving data from FAISS database: {str(e)}")
            raise e
