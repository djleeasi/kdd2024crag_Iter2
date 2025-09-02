from openai_api.openai_models import ModelSpec, ApiFamily
from pydantic import BaseModel
from typing import Optional
from openai import APIConnectionError, OpenAI, RateLimitError, APIError
import time
import random
from loguru import logger
from datetime import datetime
import json


def attempt_api_call(
    client,
    model_spec: ModelSpec,
    *,
    system_message: str,
    task_text: str,
    response_format: Optional[BaseModel] = None,
    max_retries=10,
):
    """
    Model이 Refuse 하는 경우는 상정하지 않음.
    """
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        payload = model_spec.build_payload(
            system_message=system_message,
            task_text=task_text,
            response_format=response_format,
        )
        try:
            if model_spec.api_family == ApiFamily.RESPONSES:
                resp = client.responses.parse(  # client: openai Client. 무슨 형식인지는 나도 몰루.
                    **payload, store=False  # Don’t persist to reduce bill/overhead
                )
            else:
                resp = client.responses.create(  # https://platform.openai.com/docs/guides/structured-outputs#json-mode
                    **payload, store=False
                )
            # log_response(payload, resp.output_text)
            return resp  # Handy helper with the assistant’s text
        except (APIConnectionError, RateLimitError, APIError) as e:
            logger.warning(f"API call failed on attempt {attempt}: {e}; backing off…")
            time.sleep(backoff * (1 + 0.25 * random.random()))
            backoff = min(backoff * 2, 16)
    return None


# def log_response(messages, response, output_directory="api_responses"):
#     """Save the response from the API to a file."""
#     os.makedirs(output_directory, exist_ok=True)
#     file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S.json")
#     file_path = os.path.join(output_directory, file_name)
#     with open(file_path, "w") as f:
#         json.dump({"messages": messages, "response": response}, f)

if __name__ == "__main__":
    from dotenv import load_dotenv

    MODEL_NAME = "gpt-5-nano"
    # MODEL_NAME = "gpt-4-0125-preview"
    import os
    from prompts.templates import (
        INSTRUCTIONS,
        IN_CONTEXT_EXAMPLES,
        EvalScheme,
        EXAMPLE_TRIPLE,
    )

    load_dotenv()

    def render_task_input(*, query: str, ground_truth: str, prediction: str) -> str:
        # Keep identical task text across both APIs.
        return f"Question: {query}\nGround truth: {ground_truth}\nPrediction: {prediction}\n"

    task_text = render_task_input(
        query=EXAMPLE_TRIPLE[0],
        ground_truth=EXAMPLE_TRIPLE[1],
        prediction=EXAMPLE_TRIPLE[2],
    )
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model_spec = ModelSpec.detect_model(MODEL_NAME)
    test_result = attempt_api_call(
        openai_client,
        model_spec,
        system_message=INSTRUCTIONS + IN_CONTEXT_EXAMPLES,
        task_text=task_text,
        response_format=EvalScheme,
    )
    print(test_result)
