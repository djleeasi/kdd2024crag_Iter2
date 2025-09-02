from dotenv import load_dotenv
import pandas as pd
import os
import json
from tqdm.auto import tqdm
from tokenizer.tokenizer_utils import trim_predictions_to_max_token_length
from openai_api.api_wrapper import attempt_api_call
from openai_api.openai_models import ModelSpec
from openai_api.prompts.templates import EvalScheme, INSTRUCTIONS, IN_CONTEXT_EXAMPLES
from loguru import logger


def load_csv(path: str) -> list[list[str]]:
    df_loaded = pd.read_csv(path)
    queries = df_loaded["query"].astype(str).tolist()
    ground_truths = df_loaded["ground_truth"].astype(str).tolist()
    predictions = df_loaded["prediction"].astype(str).tolist()
    return [queries, ground_truths, predictions]


def render_task_input(query: str, ground_truth: str, prediction: str) -> str:
    return (
        f"Question: {query}\nGround truth: {ground_truth}\nPrediction: {prediction}\n"
    )


def evaluate_predictions(
    openai_client,
    model_spec: ModelSpec,
    queries,
    ground_truths,
    predictions,
):
    n_miss, n_correct, n_correct_exact = 0, 0, 0

    system_message = INSTRUCTIONS + IN_CONTEXT_EXAMPLES

    for _idx, prediction in enumerate(
        tqdm(predictions, total=len(predictions), desc="Evaluating Predictions")
    ):
        query = queries[_idx]
        ground_truth = ground_truths[_idx].strip()
        # trim prediction to 75 tokens using Llama2 tokenizer
        prediction = trim_predictions_to_max_token_length(prediction)
        prediction = prediction.strip()

        ground_truth_lowercase = ground_truth.lower()
        prediction_lowercase = prediction.lower()

        if "i don't know" in prediction_lowercase:
            n_miss += 1
            continue
        elif prediction_lowercase == ground_truth_lowercase:
            n_correct_exact += 1
            n_correct += 1
            continue
        task_text = render_task_input(query, ground_truth, prediction)
        response = attempt_api_call(
            openai_client,
            model_spec,
            system_message=system_message,
            task_text=task_text,
            response_format=EvalScheme,
        )
        if response:
            eval_res = parse_response(response)
            if eval_res:
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


def parse_response(resp_obj) -> bool:
    """응답이 refusal인 경우는 고려하지 않음(Meta에서 제공한 Starter kit 원본 코드에서도 고려 안했음.)"""
    try:
        resp = resp_obj.output_text.lower()
        model_resp = json.loads(resp)
        answer = False
        if "accuracy" in model_resp and (
            (model_resp["accuracy"] is True)
            or (
                isinstance(model_resp["accuracy"], str)
                and model_resp["accuracy"].lower() == "true"
            )
        ):
            answer = True
        else:
            raise ValueError(f"Could not parse answer from response: {model_resp}")

        return answer
    except:
        return False


if __name__ == "__main__":
    from openai import OpenAI

    # default: gpt-4-0125-preview(EXPENSIVE!)
    load_dotenv()
    EVALUATION_MODEL_NAME = os.getenv("EVALUATION_MODEL_NAME", "gpt-5-mini")
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    model_spec = ModelSpec.detect_model(EVALUATION_MODEL_NAME)
    triples = load_csv("generated/2025_08_30_crag_test_results.csv")
    evaluation_results = evaluate_predictions(
        openai_client, model_spec, triples[0], triples[1], triples[2]
    )

    # evaluation_results = evaluate_predictions(
    #     openai_client, model_spec, ["can you tell me the birthday of fanny howe?"], ["1940-10-15"], ["Fanny Howe was born on October 15, 1940."]
    # )

    with open("results.txt", "w") as f:
        json.dump(evaluation_results, f, indent=4)
