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
        # Note: the environment variable is **HF_TOKEN** (underscore),
        # not "HF-TOKEN". This is the name you must configure in Vercel
        # or any other hosting provider.
        hf_token = os.getenv("HF_TOKEN")
        self._hf_token_is_set = bool(hf_token)
        self._client = InferenceClient(model=model_id, token=hf_token)

        # Lightweight debug log for serverless environments (e.g. Vercel).
        # This never prints the token itself, only whether it is present.
        print(
            "[SentimentService] initialised",
            f"model={model_id!r}",
            f"mode={self._mode}",
            f"hf_token_set={self._hf_token_is_set}",
        )

    @staticmethod
    def _normalise_label(raw_label: str) -> str:
        label = raw_label.upper()
        if label.startswith("NEG"):
            return "NEGATIVE"
        if label.startswith("POS"):
            return "POSITIVE"
        return label

    def _run_remote(self, text: str) -> List[Dict[str, Any]]:
        """Run inference via the remote Hugging Face Inference API.

        Any exception is caught and converted into an empty list so that
        the caller can decide how to surface the problem. This is important
        for environments like Vercel where unhandled exceptions would
        otherwise result in a generic 500 error.
        """
        try:
            outputs = self._client.text_classification(text)
        except Exception as exc:  # pragma: no cover - defensive guard
            # Degrade gracefully if the remote API is unavailable or
            # misconfigured (network issues, bad HF token, rate limits,
            # missing permissions, etc.). We log the error to the server
            # console so it is visible in Vercel logs, but we do not leak
            # sensitive details back to the browser.
            print(
                f"Hugging Face Inference API error for model {self._model_id!r}:",
                repr(exc),
            )
            # An empty list signals a backend problem to the caller.
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
        - label: POSITIVE, NEGATIVE, UNCERTAIN or ERROR
        - score: confidence score for the chosen label in [0, 1]
        - probs: mapping of labels -> probabilities for transparency
        - error (optional): human-friendly backend error description
        """

        if not text or not text.strip():
            return {"label": "UNCERTAIN", "score": 0.0, "probs": {}}

        cleaned = " ".join(text.strip().split())

        # For the Vercel-friendly deployment we always use the lightweight
        # remote backend.
        outputs = self._run_remote(cleaned)

        probs: Dict[str, float] = {}
        for item in outputs:
            # huggingface_hub >=0.22 returns TextClassificationOutputElement
            # instances; older versions may return plain dicts. Support both.
            raw_label: str
            score_val: float

            if hasattr(item, "label") and hasattr(item, "score"):
                raw_label = str(getattr(item, "label"))
                score_val = float(getattr(item, "score"))
            elif isinstance(item, dict):
                raw_label = str(item.get("label", ""))
                score_val = float(item.get("score", 0.0))
            else:
                # Unexpected element type – log and skip.
                print(
                    "Unexpected element in text_classification output:",
                    type(item),
                    repr(item),
                )
                continue

            probs[self._normalise_label(raw_label)] = score_val

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
            # If the remote backend returned nothing, surface a clearer
            # diagnostic so that the UI can show *why* predictions are
            # failing instead of silently reporting 0% confidence.
            if not self._hf_token_is_set:
                error_msg = (
                    "The backend could not contact the Hugging Face "
                    "Inference API because HF_TOKEN is not configured. "
                    "Set an environment variable named HF_TOKEN with a "
                    "valid token and redeploy the app."
                )
            else:
                error_msg = (
                    "The backend called the Hugging Face Inference API "
                    "but did not receive a valid prediction. Check your "
                    "HF_TOKEN, model name, and network access in the "
                    "server logs."
                )

            return {
                "label": "ERROR",
                "score": 0.0,
                "probs": {},
                "error": error_msg,
            }

        best_label = max(probs, key=probs.get)
        best_score = probs[best_label]

        if best_score < self._min_confidence:
            label = "UNCERTAIN"
        else:
            label = best_label

        return {"label": label, "score": best_score, "probs": probs}
