import os

import gradio as gr
from fastapi import FastAPI

from .inference import SentimentService


service = SentimentService()


def _format_explanation(label: str, score: float, probs: dict | None = None) -> str:
    """Human-friendly explanation including per-class confidence.

    The model returns scores in [0, 1]. We convert them to percentages
    with one decimal place so values close to 1.0 do not always appear
    as a hard "100%".
    """

    def fmt_pct(value: float | None) -> str:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "N/A"
        # Clamp to avoid 100.0% from tiny rounding differences
        v = max(0.0, min(0.999, v)) * 100.0
        return f"{v:.1f}%"

    parts: list[str] = []

    if label == "POSITIVE":
        parts.append(
            "✅ The model thinks this text is **POSITIVE** "
            f"with confidence **{fmt_pct(score)}**."
        )
    elif label == "NEGATIVE":
        parts.append(
            "⚠️ The model thinks this text is **NEGATIVE** "
            f"with confidence **{fmt_pct(score)}**."
        )
    else:  # UNCERTAIN or anything else
        parts.append(
            "🤔 The model is **not confident enough** to make a clear "
            f"prediction (max confidence {fmt_pct(score)}). Consider rephrasing "
            "or providing more context."
        )

    # Add per-class confidence breakdown if available
    if probs:
        pos = probs.get("POSITIVE")
        neg = probs.get("NEGATIVE")
        details: list[str] = []
        if pos is not None:
            details.append(f"POSITIVE: {fmt_pct(pos)}")
        if neg is not None:
            details.append(f"NEGATIVE: {fmt_pct(neg)}")
        if details:
            parts.append("\n\n**Per-class confidence:** " + " · ".join(details))

    return "".join(parts)


def predict_sentiment(text: str) -> tuple[str, dict]:
    """Gradio callback that returns a friendly explanation and probabilities.

    Any unexpected error is caught so that the UI shows a readable
    message instead of a generic red "Error" box.
    """
    try:
        result = service.predict(text)
        label = str(result.get("label", "UNCERTAIN"))
        score = float(result.get("score", 0.0))
        probs = result.get("probs", {}) or {}
        explanation = _format_explanation(label, score, probs)
        return explanation, probs
    except Exception as exc:  # pragma: no cover - defensive guard
        # Log to server console for debugging
        print("Error in predict_sentiment:", repr(exc))
        fallback_msg = (
            "⚠️ An internal error occurred while analyzing this text. "
            "Please try again or use different input. "
            "If this keeps happening, restart the app."
        )
        return fallback_msg, {}


EXAMPLES = [
    "I absolutely loved this product, it exceeded my expectations!",
    "This was a terrible experience and I would not recommend it.",
    "The movie was okay, some parts were good but others were boring.",
]


def build_demo() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

    css = """
    :root {
        --bg-start: #020617;
        --bg-end: #0f172a;
        --accent: #38bdf8;
        --accent-soft: rgba(56, 189, 248, 0.12);
        --accent-soft-strong: rgba(56, 189, 248, 0.22);
    }

    body {
        background: radial-gradient(circle at top, var(--bg-end), var(--bg-start) 55%, #000 100%);
    }

    .gradio-container {
        max-width: 980px !important;
        margin: 0 auto;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .app-card {
        background: rgba(15, 23, 42, 0.92);
        border-radius: 18px;
        padding: 24px 24px 28px;
        box-shadow: 0 22px 45px rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.28);
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }

    .app-card::before {
        content: "";
        position: absolute;
        inset: -40%;
        background:
            radial-gradient(circle at 10% 20%, rgba(56, 189, 248, 0.18), transparent 55%),
            radial-gradient(circle at 80% 0%, rgba(59, 130, 246, 0.16), transparent 55%),
            radial-gradient(circle at 0% 100%, rgba(96, 165, 250, 0.12), transparent 55%);
        opacity: 0.85;
        filter: blur(4px);
        z-index: -1;
        animation: float-glow 14s ease-in-out infinite alternate;
    }

    .app-header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.4rem;
        margin-bottom: 1.2rem;
    }

    .app-title {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 1.9rem;
        font-weight: 650;
        letter-spacing: 0.02em;
        color: #e5e7eb;
    }

    .app-title-accent {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        border-radius: 999px;
        background: radial-gradient(circle at 30% 10%, #facc15, #fb923c 40%, #f97316 65%, #b91c1c 100%);
        box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.8), 0 18px 35px rgba(15, 23, 42, 0.9);
        font-size: 1.2rem;
    }

    .app-subtitle {
        margin-top: 0.3rem;
        color: #cbd5f5;
        font-size: 0.95rem;
    }

    .app-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.75rem;
    }

    .app-pill {
        border-radius: 999px;
        padding: 4px 11px;
        background: var(--accent-soft);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #e0f2fe;
        font-size: 0.76rem;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }

    .app-credit {
        margin-top: 0.65rem;
        font-size: 0.9rem;
        color: #c4d3ff;
    }

    .emoji-orbit {
        position: relative;
        width: 170px;
        height: 170px;
        margin: 0 auto 0.75rem auto;
        border-radius: 999px;
        background: radial-gradient(circle at 30% 10%, rgba(251, 191, 36, 0.22), transparent 65%),
                    radial-gradient(circle at 70% 90%, rgba(56, 189, 248, 0.22), transparent 65%);
        border: 1px solid rgba(148, 163, 184, 0.55);
        box-shadow: 0 18px 35px rgba(15, 23, 42, 0.8);
        overflow: hidden;
    }

    .emoji-orbit-center {
        position: absolute;
        inset: 26%;
        border-radius: inherit;
        background: radial-gradient(circle at 30% 20%, #facc15, #f97316 45%, #b91c1c 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.4rem;
        filter: drop-shadow(0 10px 20px rgba(15, 23, 42, 0.9));
    }

    .emoji-orbit-ring {
        position: absolute;
        inset: 10%;
        border-radius: inherit;
        border: 1px dashed rgba(148, 163, 184, 0.85);
        opacity: 0.7;
    }

    .emoji-orbit-icon {
        position: absolute;
        font-size: 1.35rem;
        filter: drop-shadow(0 6px 10px rgba(15, 23, 42, 0.95));
        animation: orbit 16s linear infinite;
        transform-origin: 50% 50%;
    }

    .emoji-orbit-icon.positive {
        top: 6%;
        left: 63%;
    }

    .emoji-orbit-icon.negative {
        bottom: -2%;
        right: 4%;
        animation-delay: -4s;
    }

    .emoji-orbit-icon.neutral {
        top: 22%;
        left: -2%;
        animation-delay: -8s;
    }

    .analyze-button {
        font-weight: 600;
        border-radius: 999px !important;
        box-shadow: 0 14px 28px rgba(56, 189, 248, 0.35);
        transition: transform 0.16s ease-out, box-shadow 0.16s ease-out, filter 0.16s ease-out;
        animation: pulse 2.7s ease-in-out infinite;
    }

    .analyze-button:hover {
        transform: translateY(-1px) scale(1.01);
        filter: brightness(1.05);
        box-shadow: 0 18px 32px rgba(56, 189, 248, 0.45);
    }

    .analyze-button:active {
        transform: translateY(0px) scale(0.99);
        box-shadow: 0 10px 20px rgba(56, 189, 248, 0.35);
    }

    .char-counter {
        font-size: 0.8rem;
        color: #9ca3af;
        text-align: right;
        margin-top: 4px;
    }

    .fade-in {
        animation: fade-in-up 0.7s ease-out;
    }

    .app-footer {
        margin-top: 1.2rem;
        font-size: 0.83rem;
        color: #9ca3af;
    }

    .app-footer strong {
        color: #e5e7eb;
    }

    @keyframes float-glow {
        0% { transform: translate3d(-6px, 8px, 0) scale(1.02); opacity: 0.85; }
        50% { transform: translate3d(12px, -10px, 0) scale(1.06); opacity: 1; }
        100% { transform: translate3d(6px, 4px, 0) scale(1.02); opacity: 0.9; }
    }

    @keyframes pulse {
        0%, 100% { box-shadow: 0 14px 26px rgba(56, 189, 248, 0.28); }
        50% { box-shadow: 0 20px 38px rgba(56, 189, 248, 0.52); }
    }

    @keyframes orbit {
        0% { transform: rotate(0deg) translateX(6px) rotate(0deg); }
        50% { transform: rotate(180deg) translateX(6px) rotate(-180deg); }
        100% { transform: rotate(360deg) translateX(6px) rotate(-360deg); }
    }

    @keyframes fade-in-up {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    """

    with gr.Blocks(theme=theme, css=css, title="Sentiment Analysis Assistant") as demo:
        with gr.Column(elem_classes=["app-card"]):
            gr.HTML(
                """
                <div class="app-header-row">
                  <div>
                    <div class="app-title">
                      <div class="app-title-accent">💬</div>
                      <span>Sentiment Analysis Assistant</span>
                    </div>
                    <p class="app-subtitle">
                      Understand the emotional tone of any English sentence using a
                      BERT-based sentiment model with confidence-aware predictions.
                    </p>
                    <div class="app-pill-row">
                      <span class="app-pill">⚡ Real-time, browser-based experience</span>
                      <span class="app-pill">🧠 DistilBERT sentiment pipeline</span>
                      <span class="app-pill">🛡️ UNCERTAIN for low confidence</span>
                    </div>
                    <p class="app-credit">
                      Crafted with care by <strong>Shivansh Mishra</strong>.
                    </p>
                  </div>
                </div>
                """
            )

            with gr.Row():
                with gr.Column(scale=2, min_width=420):
                    text_input = gr.Textbox(
                        lines=5,
                        label="Your text",
                        elem_id="input-text",
                        placeholder="Type any sentence in English, e.g. 'I really enjoyed this movie.'",
                    )
                    gr.HTML(
                        """
                        <div id="char-counter" class="char-counter">0 characters</div>
                        <script>
                        (function() {
                          function updateCount() {
                                                        var counter = document.getElementById('char-counter');
                                                        if (!counter) return;
                                                        var el = document.querySelector('#input-text textarea, #input-text input, #input-text');
                                                        if (!el) return;
                                                        var len = (el.value || '').length;
                            counter.textContent = len + ' character' + (len === 1 ? '' : 's');
                          }
                                                    document.addEventListener('input', function () {
                                                        updateCount();
                                                    });
                          setTimeout(updateCount, 80);
                        })();
                        </script>
                        """
                    )
                    analyze_btn = gr.Button(
                        "Analyze sentiment",
                        variant="primary",
                        elem_classes=["analyze-button"],
                    )
                    # NOTE: We avoid Gradio's Examples component here because
                    # its default CSV-based caching relies on multiprocessing
                    # locks, which are not supported in some serverless
                    # environments (such as Vercel's Python runtime). Instead,
                    # we show static example prompts.
                    gr.Markdown(
                        """**Examples you can try:**

- I absolutely loved this product, it exceeded my expectations!
- This was a terrible experience and I would not recommend it.
- The movie was okay, some parts were good but others were boring.
"""
                    )

                with gr.Column(scale=1, min_width=320, elem_classes=["fade-in"]):
                    gr.HTML(
                        """
                        <div class="emoji-orbit" aria-hidden="true">
                          <div class="emoji-orbit-ring"></div>
                          <div class="emoji-orbit-center">🙂</div>
                          <div class="emoji-orbit-icon positive">😊</div>
                          <div class="emoji-orbit-icon negative">😟</div>
                          <div class="emoji-orbit-icon neutral">🤔</div>
                        </div>
                        """
                    )
                    explanation = gr.Markdown(label="Explanation")
                    scores = gr.Label(
                        num_top_classes=3,
                        label="Model confidence by class",
                    )

            analyze_btn.click(
                predict_sentiment,
                inputs=text_input,
                outputs=[explanation, scores],
            )
            text_input.submit(
                predict_sentiment,
                inputs=text_input,
                outputs=[explanation, scores],
            )

            gr.Markdown(
                """---
**Note:** This model is not perfect and may still be wrong,
especially for very short, ambiguous, or out‑of‑domain text. Use the
confidence scores and the UNCERTAIN category as a guide, not as a
source of absolute truth.

**Made by Shivansh Mishra.**
"""
            )

    return demo


# Global ASGI app for deployment platforms (e.g. Vercel)
# Vercel looks for a top-level `app` object in files such as `src/app.py`.
_demo = build_demo()
_fastapi_app = FastAPI()

# Mount the Gradio Blocks app onto a FastAPI application so that
# the exported `app` is a standard ASGI-compatible FastAPI instance,
# matching Vercel's documented Python/ASGI pattern.
app = gr.mount_gradio_app(_fastapi_app, _demo, path="/")


@_fastapi_app.get("/health")
async def _healthcheck() -> dict:
    return {"status": "ok"}


def main() -> None:
    port = int(os.getenv("PORT", 7860))
    # Reuse the globally constructed demo when running locally.
    _demo.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    main()
