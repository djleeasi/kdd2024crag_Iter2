import bz2
import json
import os
import time
import random

import pandas as pd

from datetime import datetime

from loguru import logger
from openai import APIConnectionError, OpenAI, RateLimitError, BadRequestError, APIError
from prompts.templates import IN_CONTEXT_EXAMPLES, INSTRUCTIONS
from tqdm.auto import tqdm
from transformers import LlamaTokenizerFast

from dotenv import load_dotenv

load_dotenv()

tokenizer = LlamaTokenizerFast.from_pretrained("tokenizer")


def load_json_file(file_path):
    """Load and return the content of a JSON file."""
    logger.info(f"Loading JSON from {file_path}")
    with open(file_path) as f:
        return json.load(f)


def get_system_message():
    """Returns the system message containing instructions and in context examples."""
    return INSTRUCTIONS + IN_CONTEXT_EXAMPLES




def attempt_api_call(client, model_name, messages, *, want_json=True,
                     use_reasoning=True, max_retries=10):
    """
    Responses API version that supports GPT-5 reasoning + strict JSON.
    - messages: same list you used before ([{"role":"system"/"user"/"assistant", ...}, ...])
    - want_json: enforce JSON output using a schema
    - use_reasoning: set minimal reasoning effort for GPT-5 models
    """
    # Strict JSON via JSON Schema (more reliable than {type:"json_object"})
    # If you know the exact schema, put it in here.
    text_cfg = None
    if want_json:
        text_cfg = {
            "format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "generic_json",
                    "strict": True,
                    "schema": {"type": "object"}  # relax or replace with your real schema
                }
            }
        }

    # Minimal reasoning keeps latency/price low
    reasoning_cfg = {"effort": "minimal"} if use_reasoning else None

    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.create(
                model=model_name,
                input=messages,                 # Responses API accepts chat-style messages
                reasoning=reasoning_cfg,        # Ignored by models that don't support it
                text=text_cfg,                  # Structured JSON output
                # max_output_tokens=max_output_tokens,
                store=False                     # Don’t persist to reduce bill/overhead
            )
            return resp.output_text            # Handy helper with the assistant’s text
        except BadRequestError as e:
            # Some non-reasoning models may reject the 'reasoning' field — fall back once
            if use_reasoning:
                logger.warning("Model rejected 'reasoning'; retrying without it once. %s", e)
                use_reasoning = False
                reasoning_cfg = None
                continue
            raise
        except (APIConnectionError, RateLimitError, APIError) as e:
            logger.warning(f"API call failed on attempt {attempt}: {e}; backing off…")
            time.sleep(backoff * (1 + 0.25 * random.random()))
            backoff = min(backoff * 2, 16)
    return None


def log_response(messages, response, output_directory="api_responses"):
    """Save the response from the API to a file."""
    os.makedirs(output_directory, exist_ok=True)
    file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S.json")
    file_path = os.path.join(output_directory, file_name)
    with open(file_path, "w") as f:
        json.dump({"messages": messages, "response": response}, f)


def parse_response(resp: str):
    """Pass auto-eval output from the evaluator."""
    try:
        resp = resp.lower()
        model_resp = json.loads(resp)
        answer = -1
        if "accuracy" in model_resp and (
            (model_resp["accuracy"] is True)
            or (
                isinstance(model_resp["accuracy"], str)
                and model_resp["accuracy"].lower() == "true"
            )
        ):
            answer = 1
        else:
            raise ValueError(f"Could not parse answer from response: {model_resp}")

        return answer
    except:
        return -1


def trim_predictions_to_max_token_length(prediction):
    """Trims prediction output to 75 tokens using Llama2 tokenizer"""
    max_token_length = 75
    tokenized_prediction = tokenizer.encode(prediction)
    trimmed_tokenized_prediction = tokenized_prediction[1 : max_token_length + 1]
    trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
    return trimmed_prediction


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
        """ Helper function to create an empty batch. """
        return {"interaction_id": [], "query": [], "search_results": [], "query_time": [], "answer": []}

    try:
        with bz2.open(dataset_path, "rt") as file:
            batch = initialize_batch()
            for line in file:
                try:
                    item = json.loads(line) # list of dictionaries, each being a question
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

    for batch in tqdm(load_data_in_batches(dataset_path, batch_size), desc="Generating predictions"):
        batch_ground_truths = batch.pop("answer")  # Remove answers from batch and store them
        batch_predictions = participant_model.batch_generate_answer(batch)
        
        queries.extend(batch["query"])
        ground_truths.extend(batch_ground_truths)
        predictions.extend(batch_predictions)
    
    return queries, ground_truths, predictions


def evaluate_predictions(queries, ground_truths, predictions, evaluation_model_name, openai_client):
    n_miss, n_correct, n_correct_exact = 0, 0, 0
    system_message = get_system_message()

    for _idx, prediction in enumerate(tqdm(
        predictions, total=len(predictions), desc="Evaluating Predictions"
    )):
        query = queries[_idx]
        ground_truth = ground_truths[_idx].strip()
        # trim prediction to 75 tokens using Llama2 tokenizer
        prediction = trim_predictions_to_max_token_length(prediction)
        prediction = prediction.strip()
        
        ground_truth_lowercase = ground_truth.lower()
        prediction_lowercase = prediction.lower()
        
        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n",
            },
        ]
        if "i don't know" in prediction_lowercase:
            n_miss += 1
            continue
        elif prediction_lowercase == ground_truth_lowercase:
            n_correct_exact += 1
            n_correct += 1
            continue

        response = attempt_api_call(openai_client, evaluation_model_name, messages)
        if response:
            log_response(messages, response)
            eval_res = parse_response(response)
            if eval_res == 1:
                n_correct += 1

    n = len(predictions)
    results = {
        "score": (2 * n_correct + n_miss) / n - 1,
        "exact_accuracy": n_correct_exact / n,
        "accuracy": n_correct / n,
        "hallucination": (n - n_correct - n_miss) / n,
        "missing": n_miss / n,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_correct_exact": n_correct_exact,
        "total": n,
    }
    logger.info(results)
    return results


if __name__ == "__main__":
    import sys
    from models.user_config import UserModel

    DATASET_PATH = "/workspace/data/6b358135-18eb-4ef1-b4c2-aa2f4e987453_crag_task_1_v0.json.bz2"
    EVALUATION_MODEL_NAME = os.getenv("EVALUATION_MODEL_NAME", "gpt-4-0125-preview")

    if os.getenv("RUN_FLAG") == "test":
        # for batch in tqdm(load_data_in_batches(DATASET_PATH, 10), desc="Testing Batches"):
        #     batch_ground_truths = batch.pop("answer")  # Remove answers from batch and store them
        sys.exit(0)

    # Generate predictions
    participant_model = UserModel()
    queries, ground_truths, predictions = generate_predictions(DATASET_PATH, participant_model)

    # Save
    df = pd.DataFrame({
    "query": queries,
    "ground_truth": ground_truths,
    "prediction": predictions
    })

    # Save as CSV
    df.to_csv("results.csv", index=False)

    # Evaluate Predictions
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    evaluation_results = evaluate_predictions(
        queries, ground_truths, predictions, EVALUATION_MODEL_NAME, openai_client
    )

    with open("results.txt", "w") as f:
        json.dump(evaluation_results, f, indent=4)

