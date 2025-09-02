import bz2
import json
import os
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv()

def load_json_file(file_path):
    """Load and return the content of a JSON file."""
    logger.info(f"Loading JSON from {file_path}")
    with open(file_path) as f:
        return json.load(f)

def load_data_in_batches(dataset_path, batch_size):
    """
    Generator function that reads data from a compressed file and yields batches of data.
    Each batch is a dictionary containing lists of interaction_ids, queries, search results, query times, and answers.

    Args:
    dataset_path (str): Path to the dataset file.
    batch_size (int): Number of data items in each batch.

    Yields:
    dict: A batch of data.
    """

    def initialize_batch():
        """Helper function to create an empty batch."""
        return {
            "interaction_id": [],
            "query": [],
            "search_results": [],
            "query_time": [],
            "answer": [],
        }

    try:
        with bz2.open(dataset_path, "rt") as file:
            batch = initialize_batch()
            for line in file:
                try:
                    item = json.loads(
                        line
                    )  # list of dictionaries, each being a question
                    for case in item:
                        for key in batch:
                            batch[key].append(case[key])
                        if len(batch["query"]) == batch_size:
                            yield batch
                            batch = initialize_batch()
                except json.JSONDecodeError:
                    logger.warn("Warning: Failed to decode a line.")
            # Yield any remaining data as the last batch
            if batch["query"]:
                yield batch
    except FileNotFoundError as e:
        logger.error(f"Error: The file {dataset_path} was not found.")
        raise e
    except IOError as e:
        logger.error(f"Error: An error occurred while reading the file {dataset_path}.")
        raise e


def generate_predictions(dataset_path, participant_model):
    """
    Processes batches of data from a dataset to generate predictions using a model.

    Args:
    dataset_path (str): Path to the dataset.
    participant_model (object): UserModel that provides `get_batch_size()` and `batch_generate_answer()` interfaces.

    Returns:
    tuple: A tuple containing lists of queries, ground truths, and predictions.
    """
    queries, ground_truths, predictions = [], [], []
    batch_size = participant_model.get_batch_size()

    for batch in tqdm(
        load_data_in_batches(dataset_path, batch_size), desc="Generating predictions"
    ):
        batch_ground_truths = batch.pop(
            "answer"
        )  # Remove answers from batch and store them
        batch_predictions = participant_model.batch_generate_answer(batch)

        queries.extend(batch["query"])
        ground_truths.extend(batch_ground_truths)
        predictions.extend(batch_predictions)

    return queries, ground_truths, predictions


if __name__ == "__main__":
    import sys
    from models.user_config import UserModel

    DATASET_PATH = (
        "/workspace/data/6b358135-18eb-4ef1-b4c2-aa2f4e987453_crag_task_1_v0.json.bz2"
    )

    if os.getenv("RUN_FLAG") == "test":
        # for batch in tqdm(load_data_in_batches(DATASET_PATH, 10), desc="Testing Batches"):
        #     batch_ground_truths = batch.pop("answer")  # Remove answers from batch and store them
        sys.exit(0)

    # Generate predictions
    participant_model = UserModel()
    queries, ground_truths, predictions = generate_predictions(
        DATASET_PATH, participant_model
    )

    # Save
    df = pd.DataFrame(
        {"query": queries, "ground_truth": ground_truths, "prediction": predictions}
    )

    # Save as CSV
    df.to_csv("results.csv", index=False)
