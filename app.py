# Import the required libraries
from flask import Flask, request, jsonify, render_template, Response
import googleapiclient.discovery
import pandas as pd
import nltk
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import traceback  # For detailed exception logging
import csv
from io import StringIO

# For PDF generation using pdfkit (make sure to install pdfkit and wkhtmltopdf)
# pip install pdfkit
# and install wkhtmltopdf from https://wkhtmltopdf.org/
import pdfkit

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load dataset and train the model
df = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')

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

def get_video_id(url):
    """
    Attempts to extract the video ID from common YouTube URL formats.
    """
    if 'v=' in url:
        video_id = url.split('v=')[1].split('&')[0]
        print(f"Extracted video ID (from 'v='): {video_id}")
        return video_id
    elif 'youtu.be/' in url:
        video_id = url.split('youtu.be/')[1].split('?')[0]
        print(f"Extracted video ID (from 'youtu.be/'): {video_id}")
        return video_id
    else:
        print("Could not extract video ID from URL:", url)
        return None

def fetch_youtube_comments(video_id, api_key):
    try:
        youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
        comments = []
        next_page_token = None

        while len(comments) < 200:
            print(f"\nFetching comments... Current total: {len(comments)}")
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,  # Maximum allowed per request
                pageToken=next_page_token
            )
            response = request.execute()
            print("Response keys:", response.keys())
            
            if "items" not in response:
                print("No 'items' key in response; perhaps comments are disabled or unavailable.")
                break

            fetched_comments = [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"] for item in response["items"]]
            print(f"Fetched {len(fetched_comments)} comments in this batch.")
            comments.extend(fetched_comments)

            next_page_token = response.get('nextPageToken')
            print(f"Next page token: {next_page_token}")
            if not next_page_token:
                print("No more pages to fetch.")
                break

        print(f"\nTotal comments fetched: {len(comments)}")
        return comments[:200]
    except Exception as e:
        traceback.print_exc()
        print(f"Error fetching comments: {e}")
        return []

def get_sentiment(text):
    try:
        processed_text = preprocess_text(text)
        transformed_text = vectorizer.transform([processed_text])
        prob_positive = model.predict_proba(transformed_text)[0][1]
        return prob_positive
    except Exception as e:
        traceback.print_exc()
        print(f"Error getting sentiment: {e}")
        return 0.0

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        youtube_url = request.form['youtube_url']
        video_id = get_video_id(youtube_url)
        if video_id:
            youtube_api_key = "AIzaSyDSxFQfAW7W2HNsV4W7DfE49ocZEjqGdA4"  # Replace with your actual key
            comments = fetch_youtube_comments(video_id, youtube_api_key)
            if comments:
                comments_with_sentiment = []
                for comment in comments:
                    prob_positive = get_sentiment(comment)
                    sentiment = 1 if prob_positive >= 0.5 else 0
                    confidence = prob_positive if sentiment == 1 else (1 - prob_positive)
                    comments_with_sentiment.append({
                        'text': comment,
                        'sentiment': sentiment,
                        'confidence': confidence
                    })

                positive_comments = [c for c in comments_with_sentiment if c['sentiment'] == 1]
                negative_comments = [c for c in comments_with_sentiment if c['sentiment'] == 0]
                positive_count = len(positive_comments)
                negative_count = len(negative_comments)

                top_positive = sorted(positive_comments, key=lambda x: x['confidence'], reverse=True)[:3]
                top_negative = sorted(negative_comments, key=lambda x: x['confidence'], reverse=True)[:3]
                summary_positive = [comment['text'] for comment in top_positive]
                summary_negative = [comment['text'] for comment in top_negative]

                overall_summary = f"Out of {len(comments_with_sentiment)} analyzed comments, {positive_count} expressed positive sentiment and {negative_count} expressed negative sentiment. "
                if positive_count >= negative_count:
                    overall_summary += ("Overall, the comments lean towards a positive tone. Many users expressed their enthusiasm and satisfaction—for example, they mentioned that " +
                                        "; ".join(summary_positive) + ". ")
                    if summary_negative:
                        overall_summary += "However, a few users also raised concerns, noting that " + "; ".join(summary_negative) + "."
                else:
                    overall_summary += ("Overall, the sentiment appears to be more negative. Numerous comments reflected dissatisfaction, with users stating that " +
                                        "; ".join(summary_negative) + ". ")
                    if summary_positive:
                        overall_summary += "Nonetheless, there were also positive remarks, such as " + "; ".join(summary_positive) + "."

                highlighted_comments = [{'text': c['text'], 'color': 'green' if c['sentiment'] == 1 else 'red'} 
                                          for c in comments_with_sentiment]

                # Render the results page with an option to download the results
                return render_template('index.html', 
                    comments=highlighted_comments,
                    positive_count=positive_count,
                    negative_count=negative_count,
                    summary_positive=summary_positive,
                    summary_negative=summary_negative,
                    overall_summary=overall_summary,
                    youtube_url=youtube_url  # Pass the URL to be used in the download link
                )
            else:
                print("No comments fetched after API call.")
                return render_template('index.html', error="No comments found.")
        else:
            print("Invalid YouTube URL provided.")
            return render_template('index.html', error="Invalid YouTube URL.")
    return render_template('index.html')

@app.route('/download', methods=['GET'])
def download_results():
    """
    Allows users to download the sentiment analysis results as a CSV or PDF.
    Expects a 'youtube_url' parameter and an optional 'format' parameter (csv or pdf).
    The CSV now includes a summary section (overall summary, counts, top comments, and pie chart data)
    followed by the detailed comments.
    """
    youtube_url = request.args.get('youtube_url')
    file_format = request.args.get('format', 'csv')
    if not youtube_url:
        return "YouTube URL parameter is required.", 400

    video_id = get_video_id(youtube_url)
    if not video_id:
        return "Invalid YouTube URL.", 400

    youtube_api_key = "AIzaSyDSxFQfAW7W2HNsV4W7DfE49ocZEjqGdA4"  # Replace with your actual key
    comments = fetch_youtube_comments(video_id, youtube_api_key)
    if not comments:
        return "No comments found for the given video.", 404

    comments_with_sentiment = []
    for comment in comments:
        prob_positive = get_sentiment(comment)
        sentiment = 1 if prob_positive >= 0.5 else 0
        confidence = prob_positive if sentiment == 1 else (1 - prob_positive)
        comments_with_sentiment.append({
            'text': comment,
            'sentiment': sentiment,
            'confidence': confidence
        })

    # Compute summary details
    positive_comments = [c for c in comments_with_sentiment if c['sentiment'] == 1]
    negative_comments = [c for c in comments_with_sentiment if c['sentiment'] == 0]
    positive_count = len(positive_comments)
    negative_count = len(negative_comments)
    top_positive = sorted(positive_comments, key=lambda x: x['confidence'], reverse=True)[:3]
    top_negative = sorted(negative_comments, key=lambda x: x['confidence'], reverse=True)[:3]
    summary_positive = [comment['text'] for comment in top_positive]
    summary_negative = [comment['text'] for comment in top_negative]
    overall_summary = f"Out of {len(comments_with_sentiment)} analyzed comments, {positive_count} expressed positive sentiment and {negative_count} expressed negative sentiment. "
    if positive_count >= negative_count:
        overall_summary += ("Overall, the comments lean towards a positive tone. Many users expressed their enthusiasm and satisfaction—for example, they mentioned that " +
                            "; ".join(summary_positive) + ". ")
        if summary_negative:
            overall_summary += "However, a few users also raised concerns, noting that " + "; ".join(summary_negative) + "."
    else:
        overall_summary += ("Overall, the sentiment appears to be more negative. Numerous comments reflected dissatisfaction, with users stating that " +
                            "; ".join(summary_negative) + ". ")
        if summary_positive:
            overall_summary += "Nonetheless, there were also positive remarks, such as " + "; ".join(summary_positive) + "."

    # Create a DataFrame from the detailed comment results (if needed)
    df_results = pd.DataFrame(comments_with_sentiment)

    if file_format.lower() == 'csv':
        # Build CSV data including a summary section at the top
        output = StringIO()
        writer = csv.writer(output)
        # Write summary section
        writer.writerow(["Overall Summary", overall_summary])
        writer.writerow(["Positive Count", positive_count])
        writer.writerow(["Negative Count", negative_count])
        writer.writerow(["Top Positive Comments", "; ".join(summary_positive)])
        writer.writerow(["Top Negative Comments", "; ".join(summary_negative)])
        writer.writerow(["Pie Chart Data", f"Positive: {positive_count}, Negative: {negative_count}"])
        writer.writerow([])  # Blank row to separate summary from table
        # Write header for detailed comments
        writer.writerow(["Comment", "Sentiment", "Confidence"])
        # Write detailed comment rows
        for row in df_results.itertuples(index=False, name=None):
            writer.writerow(row)
        csv_data = output.getvalue()
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=sentiment_analysis.csv"}
        )
    elif file_format.lower() == 'pdf':
        try:
            # Create a simple HTML table from the DataFrame
            html = df_results.to_html(index=False)
            pdf_data = pdfkit.from_string(html, False)
            return Response(
                pdf_data,
                mimetype="application/pdf",
                headers={"Content-disposition": "attachment; filename=sentiment_analysis.pdf"}
            )
        except Exception as e:
            traceback.print_exc()
            return f"Error generating PDF: {str(e)}", 500
    else:
        return "Unsupported file format. Use 'csv' or 'pdf'.", 400

if __name__ == '__main__':
    app.run(debug=True)
