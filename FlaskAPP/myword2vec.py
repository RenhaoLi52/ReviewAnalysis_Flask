from joblib import load
from gensim.models import Word2Vec
import numpy as np
import json

# Load K-means and TF-IDF vectorizer
kmeans = load('./models/kmeans_model.joblib')
tfidf_vectorizer = load('./models/tfidf_vectorizer.joblib')

# Load Word2Vec models
models = [Word2Vec.load(f'./models/word2vec_model_cluster_{i}.model') for i in range(5)]

def preprocess_text(text):
    return text.lower()

def find_closest_tags(text, model, tags, topn=3):
    preprocessed_text = preprocess_text(text)
    words = preprocessed_text.split()
    valid_words = [word for word in words if word in model.wv.key_to_index]
    if not valid_words:
        return []

    valid_tags = [tag for tag in tags if tag in model.wv.key_to_index]
    tags_vectors = np.array([model.wv[tag] for tag in valid_tags])
    similarities = []
    for tag, tag_vector in zip(valid_tags, tags_vectors):
        sim = np.mean([model.wv.similarity(word, tag) for word in valid_words])
        similarities.append((tag, sim))

    closest_tags = sorted(similarities, key=lambda x: -x[1])[:topn]
    return closest_tags

def predict_tags(text, kmeans, models, tfidf_vectorizer, tags, topn=3):
    preprocessed_text = preprocess_text(text)
    tfidf_vector = tfidf_vectorizer.transform([preprocessed_text])

    cluster_label = kmeans.predict(tfidf_vector)[0]
    word2vec_model = models[cluster_label]
    return find_closest_tags(preprocessed_text, word2vec_model, tags, topn)

# Load tags
with open('Tags.json', 'r') as file:
    tags = json.load(file)['tags']
