# Step 1: Setup environment and install libraries
from google.colab import drive
drive.mount('/content/drive')

!pip install transformers

import json
import os
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# Step 2: Load data
json_path = '/content/drive/My Drive/path_to_your_json.json'

with open(json_path, 'r') as f:
    data = json.load(f)

def load_images(data, base_path='/content/drive/My Drive/rumsey/train/'):
    images = []
    labels = []
    for item in data:
        image_path = os.path.join(base_path, item['image'])
        image = Image.open(image_path).convert('RGB')
        images.append(image)
        labels.append(item['groups'][0]['text'])  # Adjust based on your JSON structure
    return images, labels

images, labels = load_images(data)

# Step 3: Prepare the model and data for training
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-stage1")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-stage1")

def preprocess(images, labels):
    return [processor(image, text, return_tensors="pt") for image, text in zip(images, labels)]

train_encodings = preprocess(images, labels)

class TextImageDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: tensor[idx] for key, tensor in self.encodings.items()}

    def __len__(self):
        return len(self.encodings)

dataset = TextImageDataset(train_encodings)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 4: Set up the training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(3):  # Number of epochs
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

# Optionally save the model
# model.save_pretrained('/content/drive/My Drive/your_model_directory')
