import numpy as np
import pandas as pd
import os
from os import walk
import seaborn as sns
import matplotlib.pyplot as plt

# Preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
nltk.download('stopwords')

# Vectorize text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Split Data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold

# Model Pipeline
from sklearn.pipeline import Pipeline
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm

# Metrics
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn import metrics
from scipy import sparse

# Check time
import time

# Save model
import joblib

from sklearn.model_selection import KFold
from sklearn.base import clone

train = pd.read_csv("train.csv")
train.head()

# print("Train datafram row count: {:d}".format(len(train)))

all_tags = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']
train['Sum'] = train[all_tags].sum(axis=1)

count = {}
for tag in all_tags:
    count[tag] = train[tag].sum(axis=0)
raw_data = {"category":list(count.keys()), "occurances": list(count.values())}

# sns.barplot(data=raw_data, x="occurances", y="category",orient="h")
# plt.show()

occur = train.groupby(['Sum']).size()
list(occur.index)
raw_data_tag_count = {"category": list(occur.index), "occurances": occur.to_list()}

# sns.barplot(data=raw_data_tag_count, x="occurances", y="category",orient="h")
# plt.ylabel('Liczba tagów', fontsize=10)
# plt.xlabel('Liczba artykułów', fontsize=10)
# plt.show()

# train.head()

train['text'] = train['TITLE'] + ' ' + train['ABSTRACT']
train.drop(columns=['TITLE','ABSTRACT','Sum'], inplace=True)

# Removing HTML tags
def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext

# Removing punctuation or special characters
def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

# Removing non-alphabetical characters
def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

# Removing stop words
stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

# Steaminig words - converting words that mean the same thing to the same word
stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

train["text"] = train["text"].str.lower()
train['text'] = train['text'].apply(cleanHtml)
train['text'] = train['text'].apply(cleanPunc)
train['text'] = train['text'].apply(keepAlpha)
train['text'] = train['text'].apply(stemming)
train['text'] = train['text'].apply(removeStopWords)

# print(train.shape)
# print(train.head())

tfidf_BR_MNB = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', BinaryRelevance(MultinomialNB())),
])
tfidf_BR_LR = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', BinaryRelevance(LogisticRegression(solver='sag'))),
])
tfidf_CC_MNB = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', ClassifierChain(MultinomialNB())),
])
tfidf_CC_LR = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', ClassifierChain(LogisticRegression(solver='sag'))),
])
tfidf_LB_MNB = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf',  LabelPowerset(MultinomialNB())),
])
tfidf_LB_LR = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf',  LabelPowerset(LogisticRegression(max_iter=120))),
])
cv_BR_MNB = Pipeline([
    ('cv', CountVectorizer(stop_words='english')),
    ('clf', BinaryRelevance(MultinomialNB())),
])
cv_BR_LR = Pipeline([
    ('cv', CountVectorizer(stop_words='english')),
    ('clf', BinaryRelevance(LogisticRegression(solver='sag'))),
])
cv_CC_MNB = Pipeline([
    ('tfidf', CountVectorizer(stop_words='english')),
    ('clf', ClassifierChain(MultinomialNB())),
])
cv_CC_LR = Pipeline([
    ('tfidf', CountVectorizer(stop_words='english')),
    ('clf', ClassifierChain(LogisticRegression(solver='sag'))),
])
cv_LP_MNB = Pipeline([
    ('tfidf', CountVectorizer(stop_words='english')),
    ('clf',  LabelPowerset(MultinomialNB())),
])
cv_LP_LR = Pipeline([
    ('tfidf', CountVectorizer(stop_words='english')),
    ('clf',  LabelPowerset(LogisticRegression(max_iter=120))),
])
cv_LP_SVC_lin = Pipeline([
    ('tfidf', CountVectorizer(stop_words='english')),
    ('clf',  (svm.SVC(kernel='linear', C=1, decision_function_shape='ovo'))),
])

CLASSIFIERS = [
    tfidf_BR_MNB,   # fast
    # tfidf_BR_LR,
    tfidf_CC_MNB,   # fast
    # tfidf_CC_LR,
    tfidf_LB_MNB,   # fastest
    # tfidf_LB_LR,
    # cv_BR_MNB,
    # cv_BR_LR,
    cv_CC_MNB,  # fast
    # cv_CC_LR,
    # cv_LP_MNB,
    # cv_LP_LR,
    # cv_LP_SVC_lin
]

CLASSIFIERS_NAMES = [
    'tfidf_BR_MNB',   # fast
    # 'tfidf_BR_LR',
    'tfidf_CC_MNB',   # fast
    # 'tfidf_CC_LR',
    'tfidf_LB_MNB',   # fastest
    # 'tfidf_LB_LR',
    # 'cv_BR_MNB',
    # 'cv_BR_LR',
    'cv_CC_MNB',  # fast
    # 'cv_CC_LR',
    # 'cv_LP_MNB',
    # 'cv_LP_LR',
    # 'cv_LP_SVC_lin'
]

kf = KFold(n_splits=5, random_state=2137, shuffle=True)
scores = np.zeros(shape=(len(CLASSIFIERS), kf.get_n_splits()))
# exit()

X = train.loc[:, "text"]
y = train.loc[:, all_tags]
split_idx = 0

for classifier_idx, clf_prot in enumerate(CLASSIFIERS):
    for train_index, test_index in kf.split(X):
        if split_idx == 5:
            split_idx = 0

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        clf = clone(clf_prot)
        time_start = time.time()
        clf.fit(X_train, y_train)
        time_end = time.time()
        predictions = clf.predict(X_test)

        print('\n' + CLASSIFIERS_NAMES[classifier_idx], ' split:', split_idx + 1)
        print('Accuracy = ', accuracy_score(y_test, predictions))
        print('F1 score is ', f1_score(y_test, predictions, average="micro"))
        print('Hamming Loss is ', hamming_loss(y_test, predictions))
        print('Time taken to fit model = ', str(time_end - time_start))

        score = accuracy_score(y_test, predictions)
        scores[classifier_idx, split_idx] = score

        split_idx += 1

print(scores.shape)
np.save("scores", scores)
