import streamlit as st
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

# Load Spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("Spacy model 'en_core_web_sm' is missing. Please check the installation.")
    nlp = None

# Load Sentence Transformer model
embedding_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

# Streamlit UI setup
st.title("Authorship Attribution System")
st.write("Analyze linguistic features and stylometry for authorship identification.")

user_input = st.text_area("Enter text for analysis:")

# Extract linguistic features using Spacy tokenizer as a fallback
def extract_linguistic_features(text):
    if nlp is not None:
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]
        pos_counts = Counter(token.pos_ for token in doc)
    else:
        sentences = text.split(".")  # Basic fallback tokenization if Spacy is unavailable
        pos_counts = {}

    words = [token.text for token in doc if token.is_alpha] if nlp else text.split()
    unique_words = set(words)

    lexical_features = {
        "Word Count": len(words),
        "Sentence Count": len(sentences),
        "Avg Word Length": np.mean([len(word) for word in words]) if words else 0,
        "Type-Token Ratio (TTR)": len(unique_words) / len(words) if words else 0
    }

    syntactic_features = {
        "Avg Sentence Length": len(words) / len(sentences) if sentences else 0,
        "Noun Usage": pos_counts.get('NOUN', 0) / len(words) if len(words) > 0 else 0,
        "Verb Usage": pos_counts.get('VERB', 0) / len(words) if len(words) > 0 else 0,
        "Adj Usage": pos_counts.get('ADJ', 0) / len(words) if len(words) > 0 else 0,
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
