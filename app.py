from flask import Flask, render_template, request, jsonify
import json
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Download required NLTK data
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# Load FAQs
with open("faqs.json", "r", encoding="utf-8") as file:
    faqs = json.load(file)

questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]


# Text preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


processed_questions = [preprocess(q) for q in questions]

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)


def get_answer(user_question):

    processed_input = preprocess(user_question)

    if not processed_input:
        return "Please enter a question."

    input_vector = vectorizer.transform([processed_input])

    similarities = cosine_similarity(
        input_vector,
        question_vectors
    )[0]

    best_match_index = similarities.argmax()
    best_score = similarities[best_match_index]

    # Minimum similarity threshold
    if best_score < 0.25:
        return "Sorry, I couldn't find a suitable answer to your question."

    return answers[best_match_index]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():

    data = request.get_json()

    user_question = data.get("question", "")

    answer = get_answer(user_question)

    return jsonify({
        "answer": answer
    })


if __name__ == "__main__":
    app.run(debug=True)