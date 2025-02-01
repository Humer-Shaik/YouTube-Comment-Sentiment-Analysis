#import the required libraries
from flask import Flask, request, jsonify, render_template
import googleapiclient.discovery
import pandas as pd
import nltk
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Download necessary NLTK data
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load dataset and train the model
df = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')

# function to prepare the data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

df['reviewText'] = df['reviewText'].astype(str).apply(preprocess_text)

# Convert text to TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['reviewText'])
y = df['Positive']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
print("Model Accuracy:", accuracy_score(y_test, model.predict(X_test)))


# function to fetch the comments from youtube using youtube api v3 engine
def fetch_youtube_comments(video_id, api_key):
    try:
        youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=50
        )
        response = request.execute()
        comments = [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"] for item in response["items"]]
        return comments
    except Exception as e:
        print(f"Error fetching comments: {e}")
        return []

# function to get sentiment of a given text using the trained model
def get_sentiment(text):
    try:
        processed_text = preprocess_text(text)
        transformed_text = vectorizer.transform([processed_text])
        prediction = model.predict(transformed_text)
        return 1 if prediction[0] == 1 else 0
    except Exception as e:
        print(f"Error getting sentiment: {e}")
        return 0

# function to highlight negative comments in the fetched comments
def highlight_negative_comments(comments):
    highlighted_comments = []
    for comment in comments:
        sentiment = get_sentiment(comment)
        highlighted_comments.append({"text": comment, "color": "red" if sentiment == 0 else "green"})
    return highlighted_comments

# function to get the vedio id from vedio url
def get_video_id(url):
    if 'v=' in url:
        return url.split('v=')[1].split('&')[0]
    return None

# Flask route for the homepage and also the main function
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        youtube_url = request.form['youtube_url']
        video_id = get_video_id(youtube_url)
        if video_id:
            youtube_api_key = "AIzaSyDSxFQfAW7W2HNsV4W7DfE49ocZEjqGdA4"
            comments = fetch_youtube_comments(video_id, youtube_api_key)
            if comments:
                highlighted_comments = highlight_negative_comments(comments)
                return render_template('index.html', comments=highlighted_comments)
            else:
                return render_template('index.html', error="No comments found.")
        else:
            return render_template('index.html', error="Invalid YouTube URL.")
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)