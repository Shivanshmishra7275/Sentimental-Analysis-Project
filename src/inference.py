import os
from typing import Dict, Mapping

import torch
from transformers import pipeline

from .config import DEFAULT_MODEL_NAME


class SentimentService:
    """Wrapper around a Hugging Face sentiment classifier.

    This class adds a small amount of safety on top of the raw model by:
    - normalising labels to POSITIVE / NEGATIVE
    - exposing full probability scores
    - marking low‑confidence predictions as UNCERTAIN instead of over‑confident
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        min_confidence: float = 0.6,
    ) -> None:
        model_id = model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
        device = 0 if torch.cuda.is_available() else -1
        self._pipeline = pipeline(
            "sentiment-analysis", model=model_id, device=device
        )
        # below this threshold we will mark the prediction as UNCERTAIN
        self._min_confidence = float(min_confidence)

    @staticmethod
    def _normalise_label(raw_label: str) -> str:
        label = raw_label.upper()
        if label.startswith("NEG"):
            return "NEGATIVE"
        if label.startswith("POS"):
            return "POSITIVE"
        return label

    def predict(self, text: str) -> Dict[str, object]:
        """Return a robust sentiment prediction for a single piece of text.

        The result dictionary contains:
        - label: POSITIVE, NEGATIVE or UNCERTAIN
        - score: confidence score for the chosen label in [0, 1]
        - probs: mapping of labels -> probabilities for transparency
        """

        if not text or not text.strip():
            return {"label": "UNCERTAIN", "score": 0.0, "probs": {}}

        cleaned = " ".join(text.strip().split())

        # Ask Transformers to return all scores so we can inspect confidence.
        raw_outputs = self._pipeline(cleaned, return_all_scores=True)

        # transformers>=5 can return either:
        # - List[Dict] for a single string input, or
        # - List[List[Dict]] when called with a batch.
        if isinstance(raw_outputs, list):
            if raw_outputs and isinstance(raw_outputs[0], dict):
                outputs = raw_outputs
            elif raw_outputs and isinstance(raw_outputs[0], list):
                outputs = raw_outputs[0]
            else:
                outputs = []
        else:
            outputs = []

        probs: Dict[str, float] = {}
        for item in outputs:
            raw_label = str(item.get("label", ""))
            score = float(item.get("score", 0.0))
            probs[self._normalise_label(raw_label)] = score

        # For this binary sentiment model, newer transformers versions may
        # return only the top class with its probability. To give users a
        # clearer picture, infer the complementary class probability so that
        # POSITIVE and NEGATIVE sum to ~1.0.
        if len(probs) == 1:
            if "POSITIVE" in probs:
                comp = max(0.0, min(1.0, 1.0 - probs["POSITIVE"]))
                probs.setdefault("NEGATIVE", comp)
            elif "NEGATIVE" in probs:
                comp = max(0.0, min(1.0, 1.0 - probs["NEGATIVE"]))
                probs.setdefault("POSITIVE", comp)

        if not probs:
            return {"label": "UNCERTAIN", "score": 0.0, "probs": {}}

        best_label = max(probs, key=probs.get)
        best_score = probs[best_label]

        if best_score < self._min_confidence:
            label = "UNCERTAIN"
        else:
            label = best_label

        return {"label": label, "score": best_score, "probs": probs}
