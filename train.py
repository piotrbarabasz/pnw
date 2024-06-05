import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
import joblib
import time
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch

# Download stopwords
nltk.download('stopwords')

# Load and prepare data
train = pd.read_csv("train.csv")
all_tags = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']
train['text'] = train['TITLE'] + ' ' + train['ABSTRACT']
train.drop(columns=['TITLE','ABSTRACT'], inplace=True)

# Define text cleaning functions
def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, ' ', str(sentence))

def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    return cleaned.strip().replace("\n", " ")

def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word + " "
    return alpha_sent.strip()

stop_words = set(stopwords.words('english'))
stop_words.update(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'may', 'also', 'across', 'among', 'beside', 'however', 'yet', 'within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)

def removeStopWords(sentence):
    return re_stop_words.sub(" ", sentence)

stemmer = SnowballStemmer("english")

def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem + " "
    return stemSentence.strip()

# Apply text cleaning
train["text"] = train["text"].str.lower()
train['text'] = train['text'].apply(cleanHtml)
train['text'] = train['text'].apply(cleanPunc)
train['text'] = train['text'].apply(keepAlpha)
train['text'] = train['text'].apply(stemming)
train['text'] = train['text'].apply(removeStopWords)

# Split data using train_test_split
X_train, X_test, y_train, y_test = train_test_split(train['text'], train[all_tags], test_size=0.2, random_state=42)

# Create a Hugging Face Dataset
def convert_labels_to_tensor(labels):
    return torch.tensor(labels, dtype=torch.float)

train_dataset = Dataset.from_dict({
    'text': X_train.tolist(),
    'labels': [convert_labels_to_tensor(label) for label in y_train.values]
})
test_dataset = Dataset.from_dict({
    'text': X_test.tolist(),
    'labels': [convert_labels_to_tensor(label) for label in y_test.values]
})

# Tokenize the dataset
tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Load model
model = BertForSequenceClassification.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', num_labels=len(all_tags))

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Define a function to compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = torch.sigmoid(torch.tensor(pred.predictions)).round()
    f1 = f1_score(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    hamming = hamming_loss(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'hamming_loss': hamming
    }

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model('trained_llm_model')
