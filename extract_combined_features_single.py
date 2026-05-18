import numpy as np
import joblib
import gensim
import spacy
import re

import os

# Load all saved vectorizers/models
tfidf_vectorizer = joblib.load("outputs/vectorizers/tfidf_vectorizer.pkl")
count_vectorizer = joblib.load("outputs/vectorizers/count_vectorizer.pkl")
frequency_vectorizer = joblib.load("outputs/vectorizers/frequency_vectorizer.pkl")

# Load Word2Vec and GloVe models
w2v_path = "outputs/models/word2vec_model.bin"
import gensim.downloader as api
if os.path.exists(w2v_path):
    word2vec_model = gensim.models.KeyedVectors.load(w2v_path, mmap='r')
else:
    print("Word2Vec model not found locally. Downloading from Gensim api...")
    word2vec_model = api.load("word2vec-google-news-300")
    os.makedirs("outputs/models", exist_ok=True)
    word2vec_model.save(w2v_path)

glove_model = api.load("glove-wiki-gigaword-300")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    import subprocess
    import sys
    print("SpaCy en_core_web_md model not found. Downloading...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_md"], check=True)
    nlp = spacy.load("en_core_web_md")

# Load StandardScaler
scaler = joblib.load("outputs/models/scaler.pkl")

# Hesitation words
hesitation_words = {'uh', 'um', 'erm', 'ah', 'eh', 'hmm'}

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower()

def get_avg_embedding(text, model, dim=300):
    words = text.split()
    valid = [w for w in words if w in model]
    return np.mean([model[w] for w in valid], axis=0) if valid else np.zeros(dim)

def count_hesitations(text):
    return sum(1 for w in text.lower().split() if w in hesitation_words)

def extract_linguistic_features(text):
    doc = nlp(text)
    num_tokens = len(doc)
    num_sentences = len(list(doc.sents))
    avg_token_length = np.mean([len(token.text) for token in doc]) if num_tokens > 0 else 0
    noun_count = sum(1 for token in doc if token.pos_ == "NOUN")
    verb_count = sum(1 for token in doc if token.pos_ == "VERB")
    return [num_tokens, num_sentences, avg_token_length, noun_count, verb_count]

def extract_features(text):
    text = clean_text(text)

    # Vectorizer-based features
    tfidf_feat = tfidf_vectorizer.transform([text]).toarray()
    count_feat = count_vectorizer.transform([text]).toarray()
    freq_feat = frequency_vectorizer.transform([text]).toarray()

    # Word2Vec, GloVe, Sentence Embedding
    word2vec_feat = get_avg_embedding(text, word2vec_model)
    glove_feat = get_avg_embedding(text, glove_model)
    sentence_embed = nlp(text).vector

    # Hesitation and linguistic
    hesitation_feat = np.array([[count_hesitations(text)]])
    linguistic_feat = np.array([extract_linguistic_features(text)])

    # Reshape embeddings
    word2vec_feat = word2vec_feat.reshape(1, -1)
    glove_feat = glove_feat.reshape(1, -1)
    sentence_embed = sentence_embed.reshape(1, -1)

    # Combine all
    combined = np.concatenate([
        tfidf_feat,
        count_feat,
        word2vec_feat,
        word2vec_feat,  # again for embed_word2vec
        glove_feat,
        sentence_embed,
        freq_feat,
        hesitation_feat,
        linguistic_feat
    ], axis=1)

    print("Combined shape:", combined.shape)
    scaled_combined = scaler.transform(combined)
    return scaled_combined
