import streamlit as st
import os
from plagiarism_detection import check_plagiarism, load_documents, preprocess_text

UPLOAD_FOLDER = "uploads/"

st.title("Plagiarism Checker")

uploaded_file = st.file_uploader("Upload a document", type=["txt", "docx", "pdf"])

if uploaded_file is not None:
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load and check plagiarism
    documents, file_names = load_documents(UPLOAD_FOLDER)
    input_text = preprocess_text(open(file_path, "r", encoding="utf-8").read())
    results = check_plagiarism(input_text, documents, file_names)

    # Display results
    for file, score, level, _, _ in results:
        st.write(f"**{file}**: {score}% match ({level})")
