# LLM Evaluation Framework

A modular, lightweight, and completely local LLM Evaluation Framework built to benchmark generated responses. 

Unlike enterprise platforms requiring paid API keys, this system executes **100% locally** using free open-source models, making it cost-effective, fast, and secure.

---

## 🚀 Key Features

* **Modular Metric Engine**: Built on a clean `BaseMetric` abstract class, making it easy to create and hook in custom metrics.
* **Traditional & Semantic NLP Metrics**:
  * **BLEU**: Word-gram overlap comparison using NLTK sentence smoothing.
  * **ROUGE (1, 2, L)**: Unigram, bigram, and longest common subsequence recall.
  * **BERTScore**: Semantic similarity comparison using local Hugging Face contextual embeddings (`distilbert-base-uncased`).
* **RAG & Safety Auditing**:
  * **Faithfulness**: Encodes and checks semantic overlap between prediction and retrieved contexts (`sentence-transformers`).
  * **Hallucination Detection**: Local NLI classifier (`cross-encoder/nli-deberta-v3-small`) checking for contradiction/entailment.
  * **Toxicity Detection**: Safety filters powered by local `detoxify` model packages.
  * **Bias Auditing**: Linguistic checks scanning for stereotypical language patterns.
* **LLM-as-a-Judge**: Evaluates responses for **Correctness, Relevance, Completeness, Clarity** using a local `google/flan-t5-base` model. Features a robust regex-based fallback parser to extract scores even if formatting deviates from standard JSON.
* **Extensible Custom Metrics**: Demonstrated via an example `KeywordMatchMetric` checking prompt list keywords.
* **Pairwise Model Comparison**: Directly compares outputs from Model A and Model B using score-threshold evaluation to compile win/tie rates.
* **Interactive Human Evaluation**: Terminal-guided rating flow prompting reviewers for correctness, relevance, and fluency scores (1–5) and exporting results.
* **Experiment Observability**: Fully integrated with **MLflow** for local run parameters, score logs, and report visualization, with optional **LangSmith** tracing.

---

## 🛠️ Technology Stack

* **Core**: Python, Pandas, NumPy, Pydantic
* **FastAPI**: Backend web framework & Swagger Interactive docs
* **Hugging Face**: Transformers (`torch`), Sentence-Transformers, BERTScore, Detoxify
* **Traditional NLP**: NLTK, Rouge-Score
* **Experiment Tracking**: MLflow, Typer (CLI), Rich (Console tables)
* **Testing**: Pytest

---

## 📁 Repository Structure

```text
LLM-Evaluation-Framework/
├── app/
│   ├── api/                  # FastAPI endpoints & schemas
│   ├── config/               # Settings & logging config
│   ├── evaluation/
│   │   ├── metrics/          # Modular metric class implementations
│   │   ├── evaluator.py      # Pipeline runner & dataset orchestrator
│   │   └── score_calculator.py # Weighted quality score normalizer
│   ├── services/
│   │   ├── dataset_loader.py # CSV/JSON validation loader
│   │   ├── report_generator.py # Console (Rich), JSON, and CSV exporter
│   │   └── experiment_tracker.py # MLflow & LangSmith coordinator
│   ├── cli.py                # Command-line entry points
│   └── main.py               # FastAPI application entry point
├── datasets/                 # Evaluation dataset templates
├── tests/                    # 43 comprehensive unit tests
├── requirements.txt          # Python dependencies
└── run.py                    # Startup scripts
```

---

## 💻 Setup & Installation

### 1. Clone & Activate Virtual Environment
```powershell
# Clone the repository
git clone https://github.com/mithun-27/LLM-Evaluation-Framework.git
cd LLM-Evaluation-Framework

# Create virtual environment
python -m venv .venv

# Activate (PowerShell)
.\.venv\Scripts\Activate.ps1
# Activate (Linux/macOS)
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ Command Line (CLI) Usage

Run evaluations directly in your terminal using the Typer CLI:

### 1. Evaluate a Dataset
Runs the evaluation pipeline against a CSV/JSON file. You can choose specific metrics using the `-m` option (or omit it to run all metrics):
```bash
python -m app.cli evaluate datasets/sample_dataset.csv -m bleu,rouge,bias,latency,cost
```

### 2. Pairwise Model Comparison
Scores two model output CSV datasets and prints a summarized win/tie rate analysis:
```bash
python -m app.cli compare datasets/model_a.csv datasets/model_b.csv --metric bertscore --threshold 0.01
```

### 3. Interactive Human Evaluation
Launches a step-by-step console rating flow scoring context-relevance and correctness, saving outputs to CSV/JSON:
```bash
python -m app.cli human-eval datasets/sample_dataset.csv
```

### 4. List All Supported Metrics
```bash
python -m app.cli metrics
```

---

## 🌐 API Server (FastAPI & Swagger)

To expose the evaluation engine as a REST API:
```bash
uvicorn app.main:app --reload
```
Navigate to **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** in your browser to run endpoints dynamically through the Interactive Swagger documentation.

---

## 📊 Run Tracking (MLflow)

To visualize metrics, parameter graphs, and download evaluation report artifacts:
```bash
mlflow ui
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser to see your run tracking history.

---

## 🧪 Running Unit Tests

The test suite validates calculations, fallback parsing, schemas, loader checks, and pipeline executions:
```bash
pytest
```
All 43 unit tests are fully passing.
