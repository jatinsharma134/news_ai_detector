AI Fake News Detector

An AI-powered system that detects fake news using a "Two-Tower" architecture (Text + Image analysis).

## 🚀 Features
- **Multimodal Analysis:** Uses DistilBERT for text and ResNet50 for images.
- **Real-Time Scraping:** Fetches news directly from URLs.
- **Confidence Score:** Gives a probability rating (0-100%) for authenticity.

## 🛠️ Tech Stack
- **AI Models:** PyTorch, Transformers (Hugging Face)
- **Web App:** Streamlit
- **Data Source:** Google Fact Check Tools API

## 💿 How to Run
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/ai-fake-news-detector`
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
