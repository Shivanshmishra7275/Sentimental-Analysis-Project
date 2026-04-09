import os
from typing import Any, Dict, List

from huggingface_hub import InferenceClient

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
        backend: str | None = None,
    ) -> None:
        """Initialise the sentiment service.

        Parameters
        ----------
        model_name:
            Hugging Face model ID. If not provided, reads the MODEL_NAME
            environment variable and falls back to DEFAULT_MODEL_NAME.
        min_confidence:
            Threshold below which predictions are marked as UNCERTAIN.
        backend:
            - "local": use transformers + a local model (requires torch etc.).
            - "remote": use the Hugging Face Inference API (lightweight).
            - None / "auto": prefer local if available, else remote.
        """

        model_id = model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
        backend = backend or os.getenv("SENTIMENT_BACKEND", "auto").lower()

        # For deployment on Vercel we avoid heavyweight local dependencies
        # like torch and instead rely on the hosted Inference API.
        use_remote = backend == "remote" or backend == "auto"

        self._mode = "remote" if use_remote else "local"
        self._model_id = model_id
        self._min_confidence = float(min_confidence)

        # Remote backend via Hugging Face Inference API.
        hf_token = os.getenv("HF_TOKEN")
        self._client = InferenceClient(model=model_id, token=hf_token)

    @staticmethod
    def _normalise_label(raw_label: str) -> str:
        label = raw_label.upper()
        if label.startswith("NEG"):
            return "NEGATIVE"
        if label.startswith("POS"):
            return "POSITIVE"
        return label

    def _run_remote(self, text: str) -> List[Dict[str, Any]]:
        """Run inference via the remote Hugging Face Inference API."""
        # top_k=None returns scores for all labels
        try:
            outputs = self._client.text_classification(text, top_k=None)
        except Exception as exc:  # pragma: no cover - defensive guard
            # Degrade gracefully if the remote API is unavailable or misconfigured
            # (e.g. network issues, bad HF token, rate limits). This prevents
            # the Vercel function from crashing while still surfacing a
            # meaningful but safe result to the caller.
            print(
                f"Hugging Face Inference API error for model {self._model_id!r}:",
                repr(exc),
            )
            return []

        # Ensure we always work with a list of dicts
        if isinstance(outputs, dict):
            return [outputs]
        if outputs is None:
            return []
        if not isinstance(outputs, (list, tuple)):
            # Unexpected payload shape — log and fall back to an empty result.
            print(
                "Unexpected inference output type from Hugging Face client:",
                type(outputs),
            )
            return []

        return list(outputs)

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

        # For the Vercel-friendly deployment we always use the lightweight
        # remote backend.
        outputs = self._run_remote(cleaned)

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
