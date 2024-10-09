from sklearn.pipeline import Pipeline

from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


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
    ('clf',  (SVC(kernel='linear', C=1, decision_function_shape='ovo'))),
])

CLASSIFIERS = [
    tfidf_BR_MNB,
    tfidf_BR_LR,
    tfidf_CC_MNB,
    tfidf_CC_LR,
    tfidf_LB_MNB,
    tfidf_LB_LR,
    cv_BR_MNB,
    cv_BR_LR,
    cv_CC_MNB,
    cv_CC_LR,
    cv_LP_MNB,
    cv_LP_LR,
    cv_LP_SVC_lin
]

CLASSIFIERS_NAMES = [
    'tfidf_BR_MNB',
    'tfidf_BR_LR',
    'tfidf_CC_MNB',
    'tfidf_CC_LR',
    'tfidf_LB_MNB',
    'tfidf_LB_LR',
    'cv_BR_MNB',
    'cv_BR_LR',
    'cv_CC_MNB',
    'cv_CC_LR',
    'cv_LP_MNB',
    'cv_LP_LR',
    'cv_LP_SVC_lin'
]