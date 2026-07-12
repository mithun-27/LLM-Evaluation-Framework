from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="LLM Evaluation Framework",
    description="Evaluate LLM responses using multiple metrics.",
    version="1.0.0",
)

# Include metric and evaluation routes
app.include_router(router)


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