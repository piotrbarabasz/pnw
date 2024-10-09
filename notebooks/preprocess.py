import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re

nltk.download('stopwords')


# Removing HTML tags
def clean_html(sentence):
    clean_r = re.compile('<.*?>')
    clean_text = re.sub(clean_r, ' ', str(sentence))
    return clean_text


# Removing punctuation or special characters
def clean_punc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned


# Removing non-alphabetical characters
def keep_alpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent


# Removing stop words
stop_words = set(stopwords.words('english'))
stop_words.update(
    ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'may', 'also', 'across',
     'among', 'beside', 'however', 'yet', 'within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)


def remove_stop_words(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)


# Stemminig words - converting words that mean the same thing to the same word
stemmer = SnowballStemmer("english")


def stemming(sentence):
    stem_sentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stem_sentence += stem
        stem_sentence += " "
    stem_sentence = stem_sentence.strip()
    return stem_sentence
