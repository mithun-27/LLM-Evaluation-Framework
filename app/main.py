from fastapi import FastAPI

app = FastAPI(
    title="LLM Evaluation Framework",
    description="Evaluate LLM responses using multiple metrics.",
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "message": "Welcome to LLM Evaluation Framework"
    }


@app.get("/health")
def health():
    return {
        "status": "running"
    }