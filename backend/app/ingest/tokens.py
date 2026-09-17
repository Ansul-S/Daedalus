"""Chunk sizes are measured with the embedding model's own tokenizer."""

from functools import lru_cache

from app.ingest.chunking import TokenCounter


@lru_cache
def token_counter(model_name: str) -> TokenCounter:
    from tokenizers import Tokenizer

    # Downloads only the tokenizer files (a few MB), not the model weights.
    tokenizer = Tokenizer.from_pretrained(model_name)
    tokenizer.no_truncation()

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return count_tokens
