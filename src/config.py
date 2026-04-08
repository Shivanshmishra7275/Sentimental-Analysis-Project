from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "sentiment-distilbert"

DEFAULT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
