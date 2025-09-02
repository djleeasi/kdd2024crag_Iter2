from transformers import LlamaTokenizerFast

_tokenizer = LlamaTokenizerFast.from_pretrained("tokenizer")

def trim_predictions_to_max_token_length(prediction):
    """Trims prediction output to 75 tokens using Llama2 tokenizer"""
    max_token_length = 75
    tokenized_prediction = _tokenizer.encode(prediction)
    trimmed_tokenized_prediction = tokenized_prediction[1 : max_token_length + 1]
    trimmed_prediction = _tokenizer.decode(trimmed_tokenized_prediction)
    return trimmed_prediction