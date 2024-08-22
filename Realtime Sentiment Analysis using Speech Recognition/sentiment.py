import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import json
import speech_recognition as sr
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Define class names (adjust if necessary)
class_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]


# Load the dataset
def load_dataset(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return pd.DataFrame(data)


# Load the dataset
file_path = "data.csv"
df = load_dataset(file_path)

# Preprocess the data
X = df['text'].tolist()
y = df['label'].tolist()

# Inspect unique labels in the dataset
unique_labels = set(y)
print(f"Unique labels in dataset: {unique_labels}")

# Check if unique labels are valid and create a label-to-ID mapping
label_to_id = {label: idx for idx, label in enumerate(class_names)}
print(f"Label to ID mapping: {label_to_id}")

# Convert labels to indices based on the mapping
try:
    y = [label_to_id[label] for label in y]
except KeyError as e:
    print(f"Error: Label {e} not found in class_names")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load pre-trained DistilBERT model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(class_names))

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Tokenize and encode the dataset
def encode_data(texts, labels, max_length=128):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels


# Encode the data
train_input_ids, train_attention_masks, train_labels = encode_data(X_train, y_train)
test_input_ids, test_attention_masks, test_labels = encode_data(X_test, y_test)

# Create DataLoader
batch_size = 32
train_data = TensorDataset(train_input_ids, train_attention_masks, train_labels)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = TensorDataset(test_input_ids, test_attention_masks, test_labels)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Fine-tune the model
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_masks, labels = batch

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_masks, labels = batch

        outputs = model(input_ids, attention_mask=attention_masks)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# Function for emotion analysis
def analyze_emotion(text):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]
        top_3_indices = torch.argsort(probabilities, descending=True)[:3]
        top_3_emotions = [(class_names[i], probabilities[i].item()) for i in top_3_indices]
    return top_3_emotions


# Initialize the recognizer
recognizer = sr.Recognizer()


# Function for speech-to-text conversion and emotion analysis
def process_speech():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            user_input = recognizer.recognize_google(audio)
            print(f"You said: {user_input}")
            top_3_emotions = analyze_emotion(user_input)
            print(f"Top 3 emotions for '{user_input}':")
            for emotion, prob in top_3_emotions:
                print(f"{emotion}: {prob:.4f}")
        except sr.UnknownValueError:
            print("Sorry, I did not understand the audio.")
        except sr.RequestError:
            print("Sorry, there was a problem with the speech recognition service.")


# Main loop for speech input
while True:
    command = input("Enter 'speak' to analyze speech input or 'quit' to exit: ")
    if command.lower() == 'quit':
        break
    elif command.lower() == 'speak':
        process_speech()
    else:
        print("Invalid command. Please enter 'speak' to analyze speech input or 'quit' to exit.")
