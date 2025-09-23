# YouTube Comment Sentiment Analysis

Analyze and summarize YouTube video comments using state-of-the-art NLP models. This web app fetches comments from any public YouTube video, classifies their sentiment (positive, negative, neutral, etc.), and provides insightful summaries and downloadable results.

---

## Features

- **Fetch Comments:** Enter any YouTube video URL to instantly retrieve up to 200 comments.
- **Sentiment Classification:** Uses cutting-edge zero-shot and neural network models to classify each comment as positive, negative, neutral, feedback, or spam.
- **Summarization:** Generates an overall summary of comment sentiments using transformer-based models.
- **Downloadable Results:** Export analysis as CSV for further research or reporting.
- **Modern UI:** Clean, responsive web interface built with Flask and custom CSS.
- **GPU Support:** Leverages GPU for faster inference when available.

---

## Tech Stack

- **Backend:** Python, Flask, Pandas, NLTK, scikit-learn, TensorFlow, Hugging Face Transformers
- **Frontend:** HTML, CSS (custom, responsive)
- **APIs:** YouTube Data API v3, Google Gemini API (for advanced summarization)
- **Deployment:** Dockerfile provided for containerization

---

## Installation

### Prerequisites
- Python 3.9+
- [YouTube Data API Key](https://console.developers.google.com/)
- (Optional) Google Gemini API Key for enhanced summaries

### Clone and Setup

```bash
git clone https://github.com/Humer-Shaik/YouTube-Comment-Sentiment-Analysis.git
cd YouTube-Comment-Sentiment-Analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### API Keys

Create a file named `Api keys.txt` and add:
```
gemini <YOUR_GEMINI_API_KEY>
youtube data key <YOUR_YOUTUBE_API_KEY>
```
Or set as environment variables if preferred.

### Run Locally

```bash
python app.py
```
or with Docker:
```bash
docker build -t yt-comment-sentiment .
docker run -p 5000:5000 yt-comment-sentiment
```

---

## Usage

1. Open `http://localhost:5000` in your browser.
2. Paste a YouTube video URL.
3. Click **Analyze** to fetch and process comments.
4. View classified sentiments and summary.
5. Download CSV results for your analysis.

---

## How It Works

- **Comment Fetching:** Uses YouTube Data API to extract comments.
- **Preprocessing:** Cleans and lemmatizes text, removes stopwords.
- **Classification:** 
  - **Zero-shot model:** `facebook/bart-large-mnli` (via Hugging Face) to classify comments into multiple categories.
  - **Fallback Neural Net:** Trained on Amazon reviews for binary sentiment, used if transformer unavailable.
- **Summarization:** Uses `distilbart-cnn-12-6` for concise summary of comment trends.
- **UI:** Flask serves a modern, mobile-friendly interface.

---

## Security

- **API Keys:** Do NOT commit your personal API keys.
- **Rate Limiting:** Protects against abuse (10 requests per minute, adjustable).

---

## Project Structure

```
YouTube-Comment-Sentiment-Analysis/
├── app.py               # Main Flask application
├── templates/
│   └── index.html       # Frontend UI
├── requirements.txt     # Python dependencies
├── Dockerfile           # Containerization instructions
├── Api keys.txt         # API keys (not for production)
└── venv/                # Python virtual environment
```

---

## Example Output

- **Sentiment Counts:** Number of positive, negative, neutral, feedback, and spam comments.
- **Summary:** "Most comments were positive about the video’s content, with a few negative remarks regarding production quality."
- **CSV Download:** Full list of comments with sentiment and confidence scores.

---



## 📜 License

MIT

---

## 👤 Author

Made by [Humer-Shaik](https://github.com/Humer-Shaik),[Yeshwanth N](https://github.com/yesh6289), [Dinesh M](https://github.com/Dinesh80744)
