import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, hamming_loss

dummy_text = [
    "This paper discusses the implications of quantum computing on modern cryptography.",
    "A new approach to financial modeling using machine learning techniques.",
    "The study of protein structures in biological systems.",
    "An analysis of statistical methods in big data applications.",
    "Exploring advancements in artificial intelligence for autonomous vehicles."
]

all_tags = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']

tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
model = BertForSequenceClassification.from_pretrained('./trained_llm_model')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

dummy_dataset = Dataset.from_dict({'text': dummy_text})
tokenized_dummy = dummy_dataset.map(tokenize_function, batched=True)
tokenized_dummy.set_format('torch', columns=['input_ids', 'attention_mask'])

def predict(model, dataset):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(dataset, batch_size=1):
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            preds = torch.sigmoid(outputs.logits).round().detach().cpu().numpy()
            predictions.append(preds)
    return np.concatenate(predictions, axis=0)

predictions = predict(model, tokenized_dummy)

for text, prediction in zip(dummy_text, predictions):
    print(f"Text: {text}")
    print("Predicted labels:")
    for tag, value in zip(all_tags, prediction):
        if value:
            print(f" - {tag}")
    print("\n")
