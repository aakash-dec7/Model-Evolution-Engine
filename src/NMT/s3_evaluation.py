import json
import math
import torch
import mlflow
import dagshub
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
from transformers import MarianTokenizer
from logger import logger
from src.NMT.model import model


class Evaluation:
    """
    Evaluates the model, logs the results to DAGsHub and MLflow,
    and saves the evaluation results locally.
    """

    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = MarianTokenizer.from_pretrained("src/NMT/artifacts/tokenizer")

        # Load the model
        self.model = model
        self.model.load_state_dict(torch.load("src/NMT/artifacts/model.pth"))
        self.model.to(self.device)  # Move model to correct device

        self.input_data = pd.read_csv("src/NMT/artifacts/test/input_data.csv", header=None)
        self.target_data = pd.read_csv(
            "src/NMT/artifacts/test/target_data.csv", header=None
        )

        # Convert DataFrames to PyTorch tensors
        self.input_data = torch.tensor(self.input_data.values, dtype=torch.long)
        self.target_data = torch.tensor(self.target_data.values, dtype=torch.long)

        self.dataloader = DataLoader(
            TensorDataset(self.input_data, self.target_data),
            batch_size=32,
            shuffle=True,
        )

        # Initialize MLflow tracking
        self.init_mlflow()

    def init_mlflow(self):
        """
        Initializes MLflow tracking with DAGsHub.
        """
        try:
            dagshub.init(
                repo_owner="aakash-dec7",
                repo_name="Model-Evolution-Engine",
                mlflow=True,
            )

            mlflow.set_tracking_uri(
                "https://dagshub.com/aakash-dec7/Model-Evolution-Engine.mlflow"
            )

            mlflow.set_experiment("Model-Evolution-Engine")

            # Generate a unique run name with a timestamp
            self.run_name = f"test--{datetime.now().strftime('%Y/%m/%d-%H:%M:%S')}"

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise e

    def get_git_commit_hash(self):
        """
        Retrieves the current Git commit hash.
        """
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
            )
        except subprocess.CalledProcessError:
            logger.error(f"Error: {str(e)}")
            return "unknown"

    def compute_bleu(
        self, reference: list[int], candidate: list[int], max_n: int = 4
    ):
        """
        Computes the BLEU score for model evaluation.
        """
        try:
            weights = [1 / max_n] * max_n
            precisions = []

            for n in range(1, max_n + 1):
                ref_ngrams = Counter(
                    tuple(reference[i : i + n]) for i in range(len(reference) - n + 1)
                )

                cand_ngrams = Counter(
                    tuple(candidate[i : i + n]) for i in range(len(candidate) - n + 1)
                )

                match_count = sum(
                    min(ref_ngrams[ng], cand_ngrams[ng]) for ng in cand_ngrams
                )

                total_count = max(len(candidate) - n + 1, 1)

                precisions.append(match_count / total_count if total_count > 0 else 0)

            # Compute brevity penalty
            reference_length, candidate_length = len(reference), len(candidate)
            brevity_penalty = (
                math.exp(1 - reference_length / candidate_length)
                if candidate_length < reference_length
                else 1
            )

            # Compute BLEU score, ensuring no zero log issue
            bleu_score = brevity_penalty * math.exp(
                sum(w * math.log(p) for w, p in zip(weights, precisions) if p > 0)
            )

            return bleu_score

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return 0.0

    def evaluate_model(self):
        """
        Evaluates the model performance using the BLEU score.
        """
        logger.info("Evaluating model...")

        self.model.eval()

        total_bleu_score = 0.0
        total_samples = 0

        with torch.no_grad():

            for input_batch, target_batch in self.dataloader:

                output_batch = self.model(
                    input_batch.to(self.device), target_batch.to(self.device)
                )

                # Remove <start> token
                output_batch = output_batch[:, 1:].argmax(dim=-1)
                target_batch = target_batch[:, 1:]

                # Compute BLEU scores for each sample
                batch_bleu_score = sum(
                    self.compute_bleu(ref, pred)
                    for ref, pred in zip(
                        target_batch.cpu().tolist(), output_batch.cpu().tolist()
                    )
                )

                total_bleu_score += batch_bleu_score
                total_samples += len(target_batch)

        avg_bleu_score = total_bleu_score / total_samples if total_samples > 0 else 0.0

        return avg_bleu_score

    def log_results(self, avg_bleu_score: float):
        """
        Logs model evaluation results to MLflow and saves metrics locally.
        """
        commit_hash = self.get_git_commit_hash()

        logger.info("Commit Hash: %s", commit_hash)

        try:
            with mlflow.start_run(run_name=self.run_name):

                mlflow.set_tag("mlflow.source.git.commit", commit_hash)
                mlflow.log_metrics({"avg_bleu_score": avg_bleu_score})

                mlflow.pytorch.log_model(self.model, "model")

                logger.info("Model logged as artifact")

            mlflow.end_run()

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise e

        metrics_file = Path("src/NMT/artifacts") / "metrics.json"
        metrics_file.write_text(
            json.dumps({"avg_bleu_score": avg_bleu_score}, indent=4)
        )

        logger.info("Evaluation complete | Avg BLEU Score: %.4f", avg_bleu_score)

    def run(self):
        """
        Executes the training pipeline.
        """
        try:
            avg_bleu_score = self.evaluate_model()
            self.log_results(avg_bleu_score)

            logger.info("Evaluation pipeline completed successfully")

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise e


if __name__ == "__main__":
    try:
        evaluation = Evaluation()
        evaluation.run()

    except Exception as e:
        raise RuntimeError("Evaluation pipeline failed!") from e
