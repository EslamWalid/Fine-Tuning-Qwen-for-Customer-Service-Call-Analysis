from fastapi import FastAPI
from pydantic import BaseModel
from run_model import load_model_and_tokenizer, extract_fields_with_model

app = FastAPI(title="Qwen Field Extraction API")

model, tokenizer = load_model_and_tokenizer()

class ExtractRequest(BaseModel):
    text: str

@app.post("/extract_fields")
def extract_fields(req: ExtractRequest):
    result = extract_fields_with_model(model, tokenizer, req.text)
    return {"fields": result}

# Run with: uvicorn api_server:app --reload --port 8000
