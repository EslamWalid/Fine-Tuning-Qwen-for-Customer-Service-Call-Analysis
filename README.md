# Qwen Fine-Tuned Field Extractor

A lightweight project to run your fine-tuned Qwen model for structured information extraction.  
It provides both a **Streamlit UI** and a **FastAPI server** interface.

---

## 🧩 Features

- Loads a fine-tuned Qwen model (`transformers` compatible)
- Extracts structured fields from unstructured text (expects JSON output)
- Supports:
  - 🖥️ **Streamlit app** for interactive use
  - ⚙️ **FastAPI server** for production/API integration
- GPU acceleration via `torch` if available

