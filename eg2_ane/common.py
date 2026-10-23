"""Shared constants and numpy-only helpers (no torch): the runtime imports only this module."""
from __future__ import annotations

import numpy as np

REPO = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"  # the revision the release was built and checked against
HF_MODEL = "cowboycoderhq/EmbeddingGemma-2-FP16-M6-ANE"
HIDDEN = 512
DIM = 768

PROMPTS = {  # from the model's config_sentence_transformers.json
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
    "qa": "task: question answering | query: ",
    "fact": "task: fact checking | query: ",
    "sts": "task: sentence similarity | query: ",
    "classification": "task: classification | query: ",
    "clustering": "task: clustering | query: ",
    "code": "task: code retrieval | query: ",
}


def host_embeds(table: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """(B, S) token ids -> (B, 512, 1, S) FP16 model input (the host-side token lookup)."""
    return np.ascontiguousarray(table[ids].transpose(0, 2, 1)[:, :, None, :])
