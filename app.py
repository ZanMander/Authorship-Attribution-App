import streamlit as st
from sentence_transformers import SentenceTransformer, util
import spacy
import nltk
import textstat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from collections import Counter
import re
import os

# Ensure necessary models are downloaded
os.system("python -m spacy download en_core_web_sm")
import nltk.data
import os

# Ensure NLTK's Punkt tokenizer is installed
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_path, quiet=True)

# Load NLP models
import spacy
import subprocess

# Ensure the Spacy model is installed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

embedding_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

# Streamlit App UI
st.set_page_config(page_title="Authorship Attribution App", layout="wide")
st.title("Authorship Attribution System")
st.write("Analyze linguistic features and stylometry for authorship identification.")

# User input
user_input = st.text_area("Enter text for analysis:")

# Function to extract linguistic features
def extract_linguistic_features(text):
    tokens = nltk.word_tokenize(text)
    sentences = nltk.sent_tokenize(text)
    doc = nlp(text)

    words = [token for token in tokens if token.isalpha()]
    unique_words = set(words)

    lexical_features = {
        "Word Count": len(words),
        "Sentence Count": len(sentences),
        "Avg Word Length": np.mean([len(word) for word in words]) if words else 0,
        "Type-Token Ratio (TTR)": len(unique_words) / len(words) if words else 0
    }

    pos_counts = Counter(token.pos_ for token in doc)
    syntactic_features = {
        "Avg Sentence Length": len(tokens) / len(sentences) if sentences else 0,
        "Noun Usage": pos_counts['NOUN'] / len(tokens) if len(tokens) > 0 else 0,
        "Verb Usage": pos_counts['VERB'] / len(tokens) if len(tokens) > 0 else 0,
        "Adj Usage": pos_counts['ADJ'] / len(tokens) if len(tokens) > 0 else 0,
    }

    readability_metrics = {
        "Flesch-Kincaid Reading Ease": textstat.flesch_reading_ease(text),
        "Automated Readability Index": textstat.automated_readability_index(text),
        "Dale-Chall Readability Score": textstat.dale_chall_readability_score(text),
    }

    return lexical_features, syntactic_features, readability_metrics

# Function to compute word embeddings
def get_word_embeddings(text):
    words = text.split()
    embeddings = embedding_model.encode(words)
    return words, embeddings

# Function to create a t-SNE visualization
def tsne_visualization(words, embeddings):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
    df["word"] = words

    fig, ax = plt.subplots()
    ax.scatter(df["x"], df["y"])
    for i, word in enumerate(df["word"]):
        ax.annotate(word, (df["x"][i], df["y"][i]))

    return fig

# Process analysis when user submits text
if st.button("Analyze") or user_input:
    if user_input:
        # Extract linguistic features
        lexical_features, syntactic_features, readability_metrics = extract_linguistic_features(user_input)

        st.subheader("Linguistic Features")
        st.write(lexical_features)

        st.subheader("Syntactic Features")
        st.write(syntactic_features)

        st.subheader("Readability Metrics")
        st.write(readability_metrics)

        # Compute embeddings and visualize them
        st.subheader("t-SNE Word Embedding Visualization")
        words, embeddings = get_word_embeddings(user_input)
        fig = tsne_visualization(words, embeddings)
        st.pyplot(fig)

    else:
        st.warning("Please enter text to analyze.")

st.sidebar.title("About")
st.sidebar.info("This app extracts linguistic features for authorship attribution.")
