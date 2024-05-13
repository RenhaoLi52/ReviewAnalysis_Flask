from flask import Flask, render_template, request, jsonify, flash
import json
from myword2vec import kmeans, tfidf_vectorizer, models, predict_tags, tags

from MyTransformer import load_model, predict_stars
model, tokenizer, device = load_model()



app = Flask(__name__)
app.secret_key = 'your_super_secret_key_here'

@app.route('/')
def home():
    return render_template('submit.html')

@app.route('/submit', methods=['POST'])
def submit_review():
    stars = request.form['stars']
    text = request.form['text']
    predicted_tags = predict_tags(text, kmeans, models, tfidf_vectorizer, tags)
    tag_message = ', '.join([tag[0] for tag in predicted_tags])
    predicted_tags_dicts = [{'tag': tag, 'score': float(score)} for tag, score in predicted_tags]
    predictions = predict_stars([text], model, tokenizer, device)

    data = {
        'stars': stars,
        'text': text,
        'predicted_tags': predicted_tags_dicts,
        'prediction': predictions
    }

    try:
        with open('reviews_Tags.json', 'r') as file:
            reviews = json.load(file)
    except FileNotFoundError:
        reviews = []

    reviews.append(data)
    with open('reviews_Tags.json', 'w') as file:
        json.dump(reviews, file, indent=4)

    flash(f'Review added successfully! Predicted Star: {predictions} Predicted tags: {tag_message}')
    return render_template('submit.html')

@app.route('/reviews')
def list_reviews():
    try:
        with open('reviews_Tags.json', 'r') as file:
            reviews = json.load(file)
    except FileNotFoundError:
        reviews = []
    return render_template('reviews.html', reviews=reviews)

@app.route('/center')
def list_review():
    try:
        with open('reviews_Tags.json', 'r') as file:
            reviews = json.load(file)
    except FileNotFoundError:
        reviews = []
    return render_template('center.html', reviews=reviews)

if __name__ == '__main__':
    app.run(debug=True)
