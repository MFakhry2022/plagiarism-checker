import os
import spacy
import numpy as np
import pandas as pd
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import warnings

warnings.simplefilter("ignore", FutureWarning)

# Load SpaCy and BERT models
nlp = spacy.load("en_core_web_sm")
bert_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./cache")

def preprocess_text(text):
    """Preprocess text using SpaCy: lemmatization, removing stopwords & punctuation."""
    doc = nlp(text.lower())  
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def load_documents(folder_path, max_docs=100):
    """Load and preprocess text from .docx files."""
    documents, file_names = [], []
    
    for file in os.listdir(folder_path)[:max_docs]:
        if file.endswith(".docx"):
            file_path = os.path.join(folder_path, file)
            try:
                doc = Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])
                preprocessed_text = preprocess_text(text)  
                documents.append((text, preprocessed_text))  
                file_names.append(file)
            except Exception as e:
                print(f"Error reading {file}: {e}")

    return documents, file_names

def compute_word2vec_similarity(input_text, documents):
    """Train Word2Vec model and compute similarity."""
    all_texts = [input_text] + [doc[1] for doc in documents]  
    tokenized_texts = [text.split() for text in all_texts]

    model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1, workers=4)

    def get_vector(text):
        words = text.split()
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(100) 

    input_vector = get_vector(input_text)
    doc_vectors = np.array([get_vector(doc[1]) for doc in documents])  

    return cosine_similarity([input_vector], doc_vectors)[0]

def compute_bert_similarity(input_text, documents, batch_size=10):
    """Compute BERT embeddings in batches."""
    similarities = []

    for i in range(0, len(documents), batch_size):
        batch_docs = [doc[1] for doc in documents[i:i+batch_size]]
        embeddings = bert_model.encode([input_text] + batch_docs, normalize_embeddings=True)

        input_embedding = embeddings[0].reshape(1, -1)
        doc_embeddings = embeddings[1:]

        batch_similarities = cosine_similarity(input_embedding, doc_embeddings)[0]
        similarities.extend(batch_similarities)

    return np.array(similarities)

def generate_pdf_report(results, output_path="plagiarism_report.pdf"):
    """Generate a PDF report with highlighted plagiarism."""
    pdf = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>Plagiarism Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))

    for file, score, level, text, matched_words in results:
        elements.append(Paragraph(f"<b>Document:</b> {file}", styles["Heading2"]))
        elements.append(Paragraph(f"<b>Similarity Score:</b> {score:.2f}%", styles["Normal"]))
        elements.append(Paragraph(f"<b>Plagiarism Level:</b> {level}", styles["Normal"]))
        elements.append(Spacer(1, 12))

        highlighted_text = ""
        for word in text.split():
            if word.lower() in matched_words:
                color = "red" if level == "High" else "orange" if level == "Medium" else "green"
                highlighted_text += f'<font color="{color}">{word}</font> '
            else:
                highlighted_text += word + " "

        elements.append(Paragraph(highlighted_text, styles["Normal"]))
        elements.append(Spacer(1, 24))

    pdf.build(elements)
    print(f"\n📄 Plagiarism report saved as: {output_path}")

def check_plagiarism(input_text, documents, file_names):
    """Check plagiarism using Word2Vec and BERT embeddings."""
    word2vec_similarities = compute_word2vec_similarity(input_text, documents)
    bert_similarities = compute_bert_similarity(input_text, documents)

    results = []
    for i, file in enumerate(file_names):
        avg_similarity = (word2vec_similarities[i] + bert_similarities[i]) / 2
        plagiarism_level = "High" if avg_similarity > 0.75 else "Medium" if avg_similarity > 0.50 else "Low"
        matched_words = set(input_text.split()) & set(documents[i][1].split())

        results.append((file, round(avg_similarity * 100, 2), plagiarism_level, documents[i][0], matched_words))

    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results

if __name__ == "__main__":
    folder_path = r"E:\SOFTWARE_PROJECTS\Plagiarism Checker Tool\documents"
    input_text = "This is the text you want to check for plagiarism."

    documents, file_names = load_documents(folder_path, max_docs=50)

    if documents:
        results = check_plagiarism(preprocess_text(input_text), documents, file_names)

        print("\n📌 Plagiarism Results:")
        for file, score, level, _, _ in results:
            print(f"{file}: {score:.2f}% match ({level})")

        generate_pdf_report(results)
    else:
        print("No documents found for plagiarism checking.")
