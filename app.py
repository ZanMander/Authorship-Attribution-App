import streamlit as st
import nltk
import os
import textstat
import spacy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from collections import Counter
from sentence_transformers import SentenceTransformer

# Set the page configuration
st.set_page_config(page_title="Authorship Attribution App", layout="wide")

# Define the local path to nltk_data and add it to nltk's data path
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Import NLTK tokenizers
from nltk.tokenize import word_tokenize, sent_tokenize

# Ensure punkt is available at runtime
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    raise RuntimeError("NLTK punkt tokenizer not found. Please ensure nltk_data is in the correct location.")

# Load the Spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("Spacy model 'en_core_web_sm' is missing. Please check the installation.")
    nlp = None

# Load the Sentence Transformer model
embedding_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

# Streamlit UI setup
st.title("Authorship Attribution System")
st.write("Analyze linguistic features and stylometry for authorship identification.")

user_input = st.text_area("Enter text for analysis:")

# Extract linguistic features
def extract_linguistic_features(text):
    tokens = word_tokenize(text)
    sentences = sent_tokenize(text)

    if nlp is not None:
        doc = nlp(text)
        pos_counts = Counter(token.pos_ for token in doc)
    else:
        pos_counts = {}

    words = [token for token in tokens if token.isalpha()]
    unique_words = set(words)

    lexical_features = {
        "Word Count": len(words),
        "Sentence Count": len(sentences),
        "Avg Word Length": np.mean([len(word) for word in words]) if words else 0,
        "Type-Token Ratio (TTR)": len(unique_words) / len(words) if words else 0
    }

    syntactic_features = {
        "Avg Sentence Length": len(tokens) / len(sentences) if sentences else 0,
        "Noun Usage": pos_counts.get('NOUN', 0) / len(tokens) if len(tokens) > 0 else 0,
        "Verb Usage": pos_counts.get('VERB', 0) / len(tokens) if len(tokens) > 0 else 0,
        "Adj Usage": pos_counts.get('ADJ', 0) / len(tokens) if len(tokens) > 0 else 0,
    }

    readability_metrics = {
        "Flesch-Kincaid Reading Ease": textstat.flesch_reading_ease(text),
        "Automated Readability Index": textstat.automated_readability_index(text),
        "Dale-Chall Readability Score": textstat.dale_chall_readability_score(text),
    }

    return lexical_features, syntactic_features, readability_metrics

# Compute word embeddings
def get_word_embeddings(text):
    words = text.split()
    embeddings = embedding_model.encode(words)
    return words, embeddings

# Create a t-SNE visualization
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

# Run analysis when text is submitted
if st.button("Analyze") or user_input:
    if user_input:
        lexical_features, syntactic_features, readability_metrics = extract_linguistic_features(user_input)

        st.subheader("Linguistic Features")
        st.write(lexical_features)

        st.subheader("Syntactic Features")
        st.write(syntactic_features)

        st.subheader("Readability Metrics")
        st.write(readability_metrics)

        words, embeddings = get_word_embeddings(user_input)
        fig = tsne_visualization(words, embeddings)
        st.subheader("t-SNE Word Embedding Visualization")
        st.pyplot(fig)
    else:
        st.warning("Please enter text to analyze.")

st.sidebar.title("About")
st.sidebar.info("This app extracts linguistic features for authorship attribution.")
