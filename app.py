import os
import traceback
import re
import string
import csv
from io import StringIO
import random  # (if needed for fallback simulation)
import torch
import nltk
import pandas as pd
from flask import Flask, request, render_template, Response
import googleapiclient.discovery
import requests  # For any additional API calls if needed
from transformers import pipeline
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_compress import Compress

# ----------------------- Gemini API Integration -----------------------
import google.generativeai as genai
# Replace with your actual Gemini API key
GENAI_API_KEY = -
genai.configure(api_key=GENAI_API_KEY)

# ----------------------- App Setup -----------------------
app = Flask(__name__, template_folder='templates')
Compress(app)
CORS(app)
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["200 per hour"])

# ----------------------- NLP & Transformer Setup -----------------------
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

# Use GPU if available; otherwise, use CPU.
device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device,
    batch_size=8  # Adjust based on available memory
)
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=device
)

# ----------------------- Data & Neural Network Setup -----------------------
# Load dataset (for backup binary model training, if needed)
df = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens])

df['reviewText'] = df['reviewText'].astype(str).apply(preprocess_text)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['reviewText'])
y = df['Positive']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# For simplicity, we use a basic neural network for backup sentiment classification.
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

model_filename = "nn_model.h5"
if os.path.exists(model_filename):
    try:
        # Use compile=False to avoid deserialization issues.
        nn_model = load_model(model_filename, compile=False)
        nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("Neural network model loaded from disk.")
    except Exception as e:
        print("Error loading model, retraining:", e)
        nn_model = Sequential()
        nn_model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
        nn_model.add(Dropout(0.2))
        nn_model.add(Dense(64, activation='relu'))
        nn_model.add(Dropout(0.2))
        nn_model.add(Dense(1, activation='sigmoid'))
        nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=3)
        nn_model.fit(X_train.toarray(), y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stop])
        nn_model.save(model_filename)
        print("Neural network model trained and saved.")
else:
    nn_model = Sequential()
    nn_model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    nn_model.add(Dropout(0.2))
    nn_model.add(Dense(64, activation='relu'))
    nn_model.add(Dropout(0.2))
    nn_model.add(Dense(1, activation='sigmoid'))
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    nn_model.fit(X_train.toarray(), y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stop])
    nn_model.save(model_filename)
    print("Neural network model trained and saved.")

loss, acc = nn_model.evaluate(X_test.toarray(), y_test, verbose=0)
print("Neural network model accuracy:", acc)

# ----------------------- Helper Functions -----------------------
def get_video_id(url):
    """Extracts the video ID from a YouTube URL."""
    if 'v=' in url:
        return url.split('v=')[1].split('&')[0]
    elif 'youtu.be/' in url:
        return url.split('youtu.be/')[1].split('?')[0]
    else:
        return None

def fetch_youtube_comments(video_id, api_key):
    """Fetches up to 200 comments from a YouTube video."""
    try:
        youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
        comments = []
        next_page_token = None
        while len(comments) < 200:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token
            ).execute()
            if "items" not in response:
                break
            batch = [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                     for item in response["items"]]
            comments.extend(batch)
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
        return comments[:200]  # Return up to 200 comments (if available)
    except Exception as e:
        traceback.print_exc()
        return []

def classify_comment_multiclass(comment):
    """
    Uses a zero-shot classification model to categorize a comment into one of several labels.
    """
    candidate_labels = ["positive", "negative", "neutral", "suggestions/feedback", "spam/promotional"]
    result = classifier(comment, candidate_labels)
    return result["labels"][0], result["scores"][0]

def batch_classify(comments, candidate_labels):
    """Processes a list of comments in batch using the classifier."""
    return classifier(comments, candidate_labels=candidate_labels)

def classify_comments_batch(comments):
    candidate_labels = ["positive", "negative", "neutral", "suggestions/feedback", "spam/promotional"]
    batch_results = batch_classify(comments, candidate_labels)
    return [{
        'text': comments[i],
        'category': res['labels'][0],
        'scores': res['scores'][0]
    } for i, res in enumerate(batch_results)]

def generate_summary(comments):
    """
    Uses the Gemini API (via the google.generativeai client) to generate a detailed summary of the provided comments.
    All (up to) 200 comments are concatenated into a prompt and sent to Gemini.
    If the Gemini call fails, a fallback summarizer (transformers pipeline) is used.
    """
    if not comments:
        return "No comments available to summarize."
    text = "\n".join(comments)
    prompt_text = f"Below are up to 200 YouTube comments:\n{text}\nPlease provide a detailed summary of these comments."
    try:
        response = genai.generate_text(
            model="gemini-2.0-flash",
            prompt=prompt_text,
        )
        # Depending on the response object, return the summary from the proper attribute.
        if hasattr(response, 'result'):
            return response.result
        elif hasattr(response, 'text'):
            return response.text
        else:
            return "No summary result returned from Gemini."
    except Exception as e:
        print("Exception in generate_summary (Gemini):", str(e))
        traceback.print_exc()
        # Fallback: Use the local summarizer from transformers.
        try:
            fallback = summarizer(prompt_text, max_length=150, truncation=True)
            fallback_summary = fallback[0]['summary_text']
            return fallback_summary + " (Fallback summarization used.)"
        except Exception as e2:
            print("Fallback summarization error:", str(e2))
            traceback.print_exc()
            return "Error generating summary from Gemini and fallback summarizer."

# ----------------------- Flask Routes -----------------------
@app.route('/', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def index():
    if request.method == 'POST':
        youtube_url = request.form['youtube_url']
        video_id = get_video_id(youtube_url)
        if not video_id:
            return render_template('index.html', error="Invalid YouTube URL")
        
        # Fetch YouTube comments using the provided API key.
        # Replace with your own YouTube Data API key
        api_key = -
        comments = fetch_youtube_comments(video_id, api_key)
        # Instead of checking for 200 comments, we perform analysis on however many are available.
        if not comments:
            return render_template('index.html', error="No comments are available to perform sentiment analysis.")
        
        classified = classify_comments_batch(comments)
        category_counts = {}
        for item in classified:
            cat = item["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        overall_summary = (
            f"Out of {len(classified)} analyzed comments, "
            f"{category_counts.get('positive', 0)} were positive, "
            f"{category_counts.get('negative', 0)} were negative, "
            f"{len(classified) - category_counts.get('positive', 0) - category_counts.get('negative', 0)} were neutral or other."
        )
        # Use Gemini (via google.generativeai) to generate the detailed summary from all comments.
        comment_summary = generate_summary(comments)
        
        highlighted_comments = [{
            'text': item['text'],
            'category': item['category'],
            'confidence': round(item['scores'], 2)
        } for item in classified]
        
        # Determine top 5 positive and top 5 negative comments
        top_positive = sorted(
            [c for c in highlighted_comments if c['category'] == 'positive'],
            key=lambda x: x['confidence'],
            reverse=True
        )[:5]
        top_negative = sorted(
            [c for c in highlighted_comments if c['category'] == 'negative'],
            key=lambda x: x['confidence'],
            reverse=True
        )[:5]
        
        return render_template('index.html', 
                               comments=highlighted_comments,
                               overall_summary=overall_summary,
                               comment_summary=comment_summary,
                               top_positive=top_positive,
                               top_negative=top_negative,
                               youtube_url=youtube_url,
                               category_counts=category_counts
                               )
    return render_template('index.html')

@app.route('/download', methods=['GET'])
def download_results():
    youtube_url = request.args.get('youtube_url')
    if not youtube_url:
        return "YouTube URL parameter is required.", 400
    video_id = get_video_id(youtube_url)
    if not video_id:
        return "Invalid YouTube URL.", 400
    # Replace with your own YouTube Data API key
    api_key = "AIzaSyDSxFQfAW7W2HNsV4W7DfE49ocZEjqGdA4"
    comments = fetch_youtube_comments(video_id, api_key)
    if not comments:
        return "No comments are available for analysis.", 404
    classified = classify_comments_batch(comments)
    category_counts = {}
    for item in classified:
        cat = item["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    overall_summary = (
        f"Out of {len(classified)} analyzed comments, "
        f"{category_counts.get('positive', 0)} were positive, "
        f"{category_counts.get('negative', 0)} were negative, "
        f"{len(classified) - category_counts.get('positive', 0) - category_counts.get('negative', 0)} were neutral or other."
    )
    df_results = pd.DataFrame(classified)
    # Only support CSV download now
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Overall Summary", overall_summary])
    for label, count in category_counts.items():
        writer.writerow([f"{label} Count", count])
    writer.writerow([])
    writer.writerow(["Comment", "Category", "Confidence"])
    for row in df_results.itertuples(index=False, name=None):
        writer.writerow(row)
    csv_data = output.getvalue()
    return Response(csv_data, mimetype="text/csv",
                    headers={"Content-disposition": "attachment; filename=sentiment_analysis.csv"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
