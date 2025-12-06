import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import DistilBertTokenizer, DistilBertModel
from PIL import Image
import requests
from newspaper import Article
from io import BytesIO

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="AI Fake News Detector", page_icon="🛡️")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same architecture (Must match training exactly)
class FakeNewsDetector(nn.Module):
    def __init__(self):
        super(FakeNewsDetector, self).__init__()
        self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.image_model = models.resnet50(pretrained=True)
        self.image_model.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(768 + 2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, image):
        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.last_hidden_state[:, 0, :]
        image_feat = self.image_model(image)
        combined = torch.cat((text_feat, image_feat), 1)
        output = self.classifier(combined)
        return output

# --- 2. LOAD THE BRAIN ---
@st.cache_resource
def load_model():
    model = FakeNewsDetector()
    try:
        # Load the weights you just trained
        model.load_state_dict(torch.load("fake_news_model.pth", map_location=DEVICE))
        model.to(DEVICE)
        model.eval() # Set to evaluation mode
        return model
    except FileNotFoundError:
        return None

model = load_model()

# Setup Preprocessing
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3. THE WEBSITE UI ---
st.title("🛡️ AI Fake News Detector")
st.markdown("### Industry-Level Multimodal Detection System")
st.write("This system uses **DistilBERT (Text)** and **ResNet50 (Image)** to analyze news for authenticity.")

# Input Section
url = st.text_input("Paste a News Article URL here:", placeholder="https://...")

if st.button("Analyze News"):
    if not model:
        st.error("❌ Model file 'fake_news_model.pth' not found. Did training finish?")
    elif not url:
        st.warning("Please paste a URL first.")
    else:
        with st.spinner("🕵️‍♂️ Scrapping content & Analyzing..."):
            try:
                # A. Scrape the URL
                article = Article(url)
                article.download()
                article.parse()
                
                text_content = article.text[:512] # Take first 500 chars
                image_url = article.top_image
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info("📝 **Text Analysis**")
                    st.write(text_content[:200] + "...")
                
                with col2:
                    st.info("🖼️ **Image Analysis**")
                    if image_url:
                        st.image(image_url, use_container_width=True)
                        # Download image for AI
                        response = requests.get(image_url)
                        img = Image.open(BytesIO(response.content)).convert('RGB')
                        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                    else:
                        st.warning("No image found. Using black placeholder.")
                        img_tensor = torch.zeros(1, 3, 224, 224).to(DEVICE)

                # B. Prepare Inputs
                inputs = tokenizer(
                    text_content, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=128, 
                    return_tensors="pt"
                )
                input_ids = inputs['input_ids'].to(DEVICE)
                mask = inputs['attention_mask'].to(DEVICE)

                # C. PREDICT
                with torch.no_grad():
                    prediction = model(input_ids, mask, img_tensor)
                    score = prediction.item()
                
                # D. Show Result
                st.divider()
                st.subheader("🔍 Analysis Result")
                
                # Interpret Score (0 = Fake, 1 = Real)
                # Note: Since our training data was small, we use a threshold
                confidence = score * 100
                
                if score > 0.5:
                    st.success(f"✅ Likely **REAL** News ({confidence:.1f}% Confidence)")
                    st.progress(score)
                else:
                    st.error(f"🚨 Likely **FAKE** News ({100-confidence:.1f}% Confidence)")
                    st.progress(score)

            except Exception as e:
                st.error(f"Error analyzing URL: {e}")