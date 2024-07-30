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
import morfeusz2

nltk.download('stopwords')

train = pd.read_csv("train.csv")
all_tags = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']
train['text'] = train['TITLE'] + ' ' + train['ABSTRACT']
train.drop(columns=['TITLE','ABSTRACT'], inplace=True)

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

# stop_words = set(stopwords.words('english'))
# stop_words.update(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'may', 'also', 'across', 'among', 'beside', 'however', 'yet', 'within'])
polish_stopwords = [
    'i', 'oraz', 'ale', 'a', 'z', 'w', 'na', 'do', 'od', 'za', 'przy', 'o', 'u', 'pod', 'nad',
    'po', 'przed', 'bez', 'dla', 'czy', 'że', 'to', 'jest', 'być', 'był', 'była', 'było', 'są',
    'się', 'sam', 'tak', 'nie', 'już', 'tylko', 'więc', 'kiedy', 'który', 'która', 'które',
    'ten', 'tamten', 'ta', 'te', 'ci', 'co', 'czyli', 'bardziej', 'mniej', 'tutaj', 'stąd',
    'wszędzie', 'gdzie', 'ktokolwiek', 'nikt', 'każdy', 'wszystko', 'nic', 'można', 'muszę',
    'musisz', 'chcę', 'chcesz', 'możesz', 'może', 'być', 'czemu', 'dlaczego', 'ponieważ', 'lecz',
    'zero', 'jeden', 'dwa', 'trzy', 'cztery', 'pięć', 'sześć', 'siedem', 'osiem', 'dziewięć', 'dziesięć',
    'może', 'także', 'przez', 'między', 'obok', 'jednak', 'jeszcze', 'w środku'
]
re_stop_words = re.compile(r"\b(" + "|".join(polish_stopwords) + ")\\W", re.I)

def removeStopWords(sentence):
    return re_stop_words.sub(" ", sentence)

# stemmer = SnowballStemmer("polish")
morf = morfeusz2.Morfeusz()

def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        analysis = morf.analyse(word)
        if analysis:
            stem = analysis[0][2][1].split(':')[0]
            stemSentence += stem + " "
        else:
            stemSentence += word + " "
    return stemSentence.strip()


train["text"] = train["text"].str.lower()
train['text'] = train['text'].apply(cleanHtml)
train['text'] = train['text'].apply(cleanPunc)
train['text'] = train['text'].apply(keepAlpha)
train['text'] = train['text'].apply(stemming)
train['text'] = train['text'].apply(removeStopWords)

X_train, X_test, y_train, y_test = train_test_split(train['text'], train[all_tags], test_size=0.2, random_state=42)

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

tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

model = BertForSequenceClassification.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', num_labels=len(all_tags))

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model('trained_llm_model_pl')
