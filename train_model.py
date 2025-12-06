import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# --- CONFIGURATION ---
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--> using device: {DEVICE}")

# --- 1. THE DATASET CLASS ---
class MultimodalDataset(Dataset):
    def __init__(self, dataframe, tokenizer, transform):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.transform = transform
        
        # Simple label mapping: converting text ratings to 0 (Fake) or 1 (Real)
        # We assume anything not explicitly "True" is "Fake" for this safety filter
        self.data['label_code'] = self.data['Actual Rating'].apply(
            lambda x: 1 if "True" in str(x) or "Correct" in str(x) else 0
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # A. Process Text
        text = str(row['clean_text'])
        text_tokens = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        
        # B. Process Image
        img_path = row['local_image_path']
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except:
            # If image fails, create a black image (fallback)
            image = torch.zeros(3, 224, 224)

        return {
            'input_ids': text_tokens['input_ids'].squeeze(0),
            'attention_mask': text_tokens['attention_mask'].squeeze(0),
            'image': image,
            'label': torch.tensor(row['label_code'], dtype=torch.float)
        }

# --- 2. THE MODEL ARCHITECTURE (TWO-TOWER) ---
class FakeNewsDetector(nn.Module):
    def __init__(self):
        super(FakeNewsDetector, self).__init__()
        
        # Tower 1: Text (DistilBERT)
        self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Tower 2: Image (ResNet50)
        self.image_model = models.resnet50(pretrained=True)
        self.image_model.fc = nn.Identity() # Remove the last layer to get raw features
        
        # Fusion Layer (Combine 768 text features + 2048 image features)
        self.classifier = nn.Sequential(
            nn.Linear(768 + 2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid() # Output probability between 0 and 1
        )

    def forward(self, input_ids, attention_mask, image):
        # Get Text Features
        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.last_hidden_state[:, 0, :] # Take the [CLS] token
        
        # Get Image Features
        image_feat = self.image_model(image)
        
        # Concatenate
        combined = torch.cat((text_feat, image_feat), 1)
        
        # Predict
        output = self.classifier(combined)
        return output

# --- 3. TRAINING LOOP ---
def train():
    # Load Data
    try:
        df = pd.read_csv("processed_training_data.csv")
    except FileNotFoundError:
        print("❌ Error: processed_training_data.csv not found.")
        return

    # Split Data (80% train, 20% test)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Image Transformations (Resize to 224x224 for ResNet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_dataset = MultimodalDataset(train_df, tokenizer, transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize Model
    model = FakeNewsDetector().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss() # Binary Cross Entropy for True/Fake
    
    print(f"🚀 Starting training on {len(train_df)} articles...")
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in train_loader:
            # Move data to GPU/CPU
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            images = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE).unsqueeze(1)
            
            # Forward Pass
            optimizer.zero_grad()
            outputs = model(input_ids, mask, images)
            
            # Loss Calculation
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

    # Save the Model
    torch.save(model.state_dict(), "fake_news_model.pth")
    print("\n✅ Training Complete. Model saved as 'fake_news_model.pth'")

if __name__ == "__main__":
    train()