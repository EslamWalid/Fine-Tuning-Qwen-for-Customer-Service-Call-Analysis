# 🧠 Qwen Fine-Tuned Field Extractor (Arabic Dataset)

This project provides a **Streamlit web interface** and a **Python script** for running a fine-tuned **Qwen model** that extracts key information fields from **Arabic text**.  
It was trained on a custom Arabic dataset for structured field extraction tasks (such as names, phone numbers, addresses, and order details).

---

## 🚀 Features

- Fine-tuned **Qwen language model** on **Arabic text**
- Extracts structured information and key-value pairs from unstructured Arabic sentences
- Runs locally or via Streamlit interface
- Detects JSON-like model outputs or falls back to smart line parsing

---

## ⚙️ Setup Instructions

1. **Clone or extract the folder:**
   ```bash
   git clone https://github.com/EslamWalid/qwen-finetune-field-extractor.git
   cd qwen-finetune-field-extractor

1. **Install dependencies:**
   ```bash

   pip install -r requirements.txt

2. **Add your fine-tuned Arabic model to the folder:**

Ensure the model and tokenizer files are in a directory (default: ./qwen-finetuned-model).

Adjust MODEL_NAME in run_model.py if your path differs.

3. **Launch Streamlit app:**
   ```bash
   streamlit run app_streamlit.py

