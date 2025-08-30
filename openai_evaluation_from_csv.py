from dotenv import load_dotenv
from local_evaluation import evaluate_predictions
import pandas as pd
import os
from openai import APIConnectionError, OpenAI, RateLimitError
import json

# default: gpt-4-0125-preview(EXPENSIVE!)
EVALUATION_MODEL_NAME = os.getenv("EVALUATION_MODEL_NAME", "gpt-4-0125-preview")

load_dotenv()

df_loaded = pd.read_csv("results.csv")

# Extract columns as lists
queries = df_loaded["query"].astype(str).tolist()
ground_truths = df_loaded["ground_truth"].astype(str).tolist()
predictions = df_loaded["prediction"].astype(str).tolist()

# --- Evaluate Predictions ---
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
evaluation_results = evaluate_predictions(
    queries, ground_truths, predictions, EVALUATION_MODEL_NAME, openai_client
)

# Save evaluation results
with open("results.txt", "w") as f:
    json.dump(evaluation_results, f, indent=4)