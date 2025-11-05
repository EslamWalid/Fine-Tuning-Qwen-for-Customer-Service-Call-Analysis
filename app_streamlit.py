import json
import streamlit as st
from run_model import load_model_and_tokenizer, extract_fields_with_model

@st.cache_resource
def load_cached_model():
    return load_model_and_tokenizer()

st.title("Qwen Fine-Tuned Field Extractor")
st.write("Upload or paste text below to extract structured fields using your fine-tuned Qwen model.")

text_input = st.text_area("Enter text:", height=200)

if st.button("Extract Fields"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        model, tokenizer = load_cached_model()
        with st.spinner("Extracting fields..."):
            result = extract_fields_with_model(model, tokenizer, text_input)
        st.subheader("Extracted Fields")
        st.json(result)
