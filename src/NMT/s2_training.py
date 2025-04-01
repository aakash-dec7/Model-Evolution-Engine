import os
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from src.NMT.model import model
from transformers import MarianTokenizer
from logger import logger


class Training:
    """
    Handles model training, including loading the tokenizer, dataset,
    setting up the model, optimizer, loss function, and data loader.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = MarianTokenizer.from_pretrained("src/NMT/artifacts/tokenizer")

        self.model = model

        # Load training data from CSV files (input and target sequences)
        self.input_data = pd.read_csv("src/NMT/artifacts/train/input_data.csv", header=None)
        self.target_data = pd.read_csv(
            "src/NMT/artifacts/train/target_data.csv", header=None
        )

        # Convert DataFrames to NumPy arrays before converting to PyTorch tensors
        self.input_data = torch.tensor(self.input_data.values, dtype=torch.long)
        self.target_data = torch.tensor(self.target_data.values, dtype=torch.long)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.dataloader = DataLoader(
            TensorDataset(self.input_data, self.target_data),
            batch_size=32,
            shuffle=True,
        )

    def train(self):
        """
        Train the model for a specified number of epochs.
        """
        num_epochs = 1

        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0

            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{num_epochs}")

            for input_batch, target_batch in progress_bar:

                input_batch, target_batch = input_batch.to(
                    self.device
                ), target_batch.to(self.device)

                # Reset gradients
                self.optimizer.zero_grad()

                output = self.model(input_batch, target_batch)

                # Reshape outputs and targets for loss calculation
                output = output[:, 1:].reshape(-1, output.shape[2])
                target_batch = target_batch[:, 1:].reshape(-1)

                # Compute loss
                loss = self.criterion(output, target_batch)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                self.optimizer.step()

                epoch_loss += loss.item()

                progress_bar.set_postfix(loss=loss.item())

            logger.info(f"Epoch {epoch + 1} completed with loss: {epoch_loss:.4f}")

    def save_model(self):
        """
        Saves the trained model's state dictionary to disk.
        """
        model_path = os.path.join("src/NMT/artifacts", "model.pth")

        # Save the model
        torch.save(self.model.state_dict(), model_path)

        logger.info(f"Seq2Seq saved successfully at: {model_path}")

    def run(self):
        """
        Executes the training pipeline.
        """
        try:
            self._train()
            self._save_model()

            logger.info("Training pipeline completed successfully")

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise e


if __name__ == "__main__":
    try:
        training = Training()
        training.run()

    except Exception as e:
        raise RuntimeError("Training pipeline failed!") from e
