# Sentiment & Colab GPU ML Project

## 0. Quick start (for anyone)

If you just want to **run** the project without reading everything first:

1. Open a terminal in this folder: `sentiment-mlops-app`.
2. Create and activate the environment (pick one):
   - Conda: `conda env create -f environment.yml && conda activate sentiment-mlops-env`
   - venv (Windows): `python -m venv .venv && .venv\\Scripts\\activate`
3. Install dependencies (for venv users): `pip install -r requirements.txt`.
4. Start the web app: `python run_app.py` and open `http://localhost:7860`.
5. Open `notebooks/01_train_colab_gpu.ipynb` and run cells from top to bottom to see training on the GPU/CPU.

You can come back to the sections below whenever you want more detail.

## 1. Project overview

This repository is a small but complete end‑to‑end machine learning project:

- **Training on Colab GPU** – a dedicated notebook shows how to mount Google Drive, use a GPU, train a neural network, and save artifacts back to Drive with almost no manual setup.
- **Production‑style inference service** – a Python package exposes a sentiment analysis service built on top of a DistilBERT model from Hugging Face.
- **Web application** – a simple Gradio UI turns the inference service into an interactive web app that you can run locally or in a free cloud environment.
- **Containerization** – a Dockerfile packages the whole application so it can run consistently across machines and cloud providers.

The goal is to mimic the work of a small, experienced technical team (ML engineer, architect, MLOps engineer, and web developer) while keeping the explanations straightforward.

---

## 2. Repository structure

- `environment.yml` – optional Conda environment definition for local development.
- `requirements.txt` – Python dependencies for the app and notebooks.
- `config/` – configuration support (extendable for more environments).
- `models/` – place to store any exported models you download from Colab or the Hub.
- `notebooks/01_train_colab_gpu.ipynb` – Colab‑oriented GPU training workflow.
- `src/config.py` – central configuration values such as the default model name.
- `src/inference.py` – sentiment analysis service using Hugging Face Transformers.
- `src/app.py` – Gradio web application wrapping the inference service.
- `Dockerfile` – container image definition for the web app.

---

## 2.1. Conceptual flow (high level)

At a high level, this project works as follows:

- You **experiment and train** a small neural network in the notebook (optionally on Colab GPU).
- You **serve predictions** using a separate, production‑oriented sentiment model through the web app.
- You can later swap in your own trained model by adapting the inference code.

This mirrors a typical MLOps setup where experimentation and serving are cleanly separated.

---

## 3. Separate environment setup

You can either use **Conda** (recommended for data science stacks) or a simple **virtualenv**.

### 3.1. Using Conda

```bash
cd sentiment-mlops-app
conda env create -f environment.yml
conda activate sentiment-mlops-env
```

### 3.2. Using venv + pip

```bash
cd sentiment-mlops-app
python -m venv .venv
.venv\\Scripts\\activate  # on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Colab GPU notebook (using the Colab extension)

The notebook is designed so that it works smoothly with the Google Colab VS Code extension or directly in the browser.

### 4.1. Opening the notebook on Colab GPU

1. Open `notebooks/01_train_colab_gpu.ipynb` in VS Code.
2. Use your **Colab extension** to run this notebook on a Colab backend.
3. In Colab, go to **Runtime → Change runtime type → Hardware accelerator: GPU**.
4. Run the cells from top to bottom.

### 4.2. What the notebook does

The notebook follows these steps:

1. **Mount Google Drive and verify GPU**
   - Mounts Drive at `/content/drive`.
   - Uses `nvidia-smi` and framework checks to confirm that a GPU is available.
2. **Install and import dependencies**
   - Installs PyTorch, NumPy, pandas, matplotlib, and scikit‑learn.
   - Imports all required libraries in a single place.
3. **Configure GPU device**
   - Selects `cuda` if a GPU is available, otherwise falls back to CPU.
   - Prints which device is being used and some basic GPU memory information.
4. **Load dataset and create splits**
   - Loads a classic binary classification dataset (breast cancer) from scikit‑learn.
   - Normalizes features with `StandardScaler`.
   - Builds train/validation/test splits and wraps them in PyTorch `DataLoader`s.
5. **Build a simple neural network**
   - Defines a small MLP in PyTorch with dropout and ReLU activations.
   - Uses cross‑entropy loss and Adam optimizer.
6. **Train on GPU and track metrics**
   - Sends mini‑batches to GPU.
   - Logs train/validation loss and accuracy across epochs.
   - Plots the curves so you can visually inspect training behaviour.
7. **Evaluate, checkpoint, and export**
   - Evaluates the model on the test set.
   - Saves the trained model weights and scaler statistics to Google Drive under
     `/content/drive/MyDrive/colab-ml-gpu-demo/`.
   - These files can be downloaded and reused in separate scripts or services.

The idea is that training is **GPU‑accelerated and effortless** once the Colab runtime is attached: you simply run cells, and the notebook handles device selection and saving results to Drive.

---

## 5. Sentiment inference service

While the Colab notebook shows a generic GPU training workflow, the production‑facing app focuses on **sentiment analysis** using a pre‑trained DistilBERT model.

### 5.1. Core design

- `src/inference.py` defines `SentimentService`, which:
  - Uses Hugging Face `pipeline("sentiment-analysis")`.
  - Loads the default model `distilbert-base-uncased-finetuned-sst-2-english` (configurable with `MODEL_NAME` env var).
  - Automatically selects GPU (`device=0`) if CUDA is available, or CPU otherwise.
  - Normalizes output labels to `POSITIVE` and `NEGATIVE` and returns a label + confidence score.

This gives you a clean separation between **model logic** and **web UI**, which is typical in production systems.

---

## 6. Web app: Gradio interface

The web interface is implemented in `src/app.py` using **Gradio**:

- Wraps `SentimentService` in a simple function that accepts raw text.
- Returns human‑readable text like `Prediction: POSITIVE (confidence: 0.98)`.
- Exposes a single‑page interface with a multi‑line textbox input and a text output.
- Can run locally or in a container.

### 6.1. Running the web app locally

With your environment activated and dependencies installed:

```bash
cd sentiment-mlops-app
python -m src.app
```

Then open the URL printed by Gradio (typically `http://127.0.0.1:7860`).

If you have a local GPU and a compatible PyTorch/Transformers install, the model will automatically use it for faster inference.

---

## 7. Dockerization (all project files inside the image)

The `Dockerfile` packages the entire project into a container image.

### 7.1. Build the image

From the `sentiment-mlops-app` directory:

```bash
docker build -t sentiment-mlops-app .
```

This copies **all project files** into `/app` inside the image, installs Python dependencies, and configures the default model.

### 7.2. Run the container

```bash
docker run --rm -p 7860:7860 sentiment-mlops-app
```

Then open `http://localhost:7860` in your browser.

The container starts the Gradio app by running:

- `python -m src.app`

If you want to change the underlying model, you can override the environment variable:

```bash
docker run --rm -p 7860:7860 -e MODEL_NAME=distilbert-base-uncased sentiment-mlops-app
```

---

## 8. Free deployment ideas

There are several free or low‑cost ways to deploy this web app:

- **Hugging Face Spaces (Gradio)**
  - Push this project (or just `src/app.py` and `requirements.txt`) to a Git repository.
  - Create a new Gradio Space and point it to that repo.
  - HF will install dependencies and run `app.py` automatically.
- **Free container platforms** (when available)
  - Use the provided `Dockerfile` to push an image to Docker Hub or a container registry.
  - Deploy it to a platform that supports free container apps.

- **Vercel Python runtime (ASGI app)**
   - Connect this GitHub repo to Vercel and set the root directory to `sentiment-mlops-app`.
   - Vercel detects `src/app.py` which exposes a top‑level ASGI application named `app` via Gradio's `App.create_app`, so no custom server or Uvicorn command is necessary.
   - Define environment variables in the Vercel project settings, especially:
      - `HF_TOKEN` – your Hugging Face Inference API token so remote sentiment inference works.
      - (Optional) `MODEL_NAME` and `SENTIMENT_BACKEND` if you want to override defaults.
   - The `.python-version` file pins Python 3.13, matching local development and ensuring a compatible runtime.

In all cases, inference is relatively lightweight and can run comfortably on CPU. GPU is mainly required for **training**, which is handled in Colab.

---

## 9. How the technical team approached this

This project was structured to resemble the work of different specialists:

- **ML Architect** – defined the split between experimental training (Colab notebook) and a stable, pre‑trained model for production sentiment analysis.
- **ML Engineer** – implemented the PyTorch training loop in the notebook, GPU device handling, and metric tracking.
- **MLOps Engineer** – introduced environment management (Conda + `requirements.txt`), a clear project layout, and a Dockerfile that encapsulates all runtime dependencies.
- **Web Developer** – built the Gradio interface in `src/app.py`, focusing on a minimal but usable UI and clear text outputs.

Key design choices and efforts:

- **Colab‑first training experience** – by handling Drive mounting, GPU checks, and saving artifacts, the notebook minimises friction when using the Colab extension.
- **Separation of concerns** – training and inference are kept decoupled so you can update one without breaking the other.
- **Reproducible environments** – the environment file and `requirements.txt` make it straightforward to recreate the setup on another machine or inside Docker.
- **Simple, portable web UI** – Gradio allows rapid iteration while still being easy to containerize and deploy.

---

## 10. Next steps and customisation

Here are natural extensions you can explore:

- Replace the breast‑cancer example in the notebook with your own dataset and architecture while keeping the same GPU workflow.
- Export a custom model from Colab and load it in a dedicated inference script (similar to `SentimentService`).
- Add logging and monitoring (e.g. request logging, basic metrics) to the web app for a more MLOps‑heavy setup.
- Integrate a CI pipeline that builds and pushes the Docker image on each commit.

This repository is intentionally small but complete, so you can treat it as a template for future end‑to‑end ML projects using Colab, web apps, and Docker.

---

## 11. FAQ / common issues

- **Port 7860 already in use**  
   Another process is using the default Gradio port. Either close the other app or run:  
   `set GRADIO_SERVER_PORT=7870` (Windows) and then `python run_app.py`.

- **No GPU found / running on CPU**  
   The notebook and web app both work on CPU. A GPU is only needed for *faster* training. If you are on Colab, make sure you selected a GPU runtime under **Runtime → Change runtime type**.

- **Model download from Hugging Face is slow**  
   The first run downloads the DistilBERT model and tokenizer. This can take a few minutes depending on your connection. Subsequent runs are much faster because the files are cached locally.

- **Colab `google.colab` import error locally**  
   The notebook automatically detects when it is not running on Colab and skips the Drive mount. You do not need to install `google.colab` on your local machine.
