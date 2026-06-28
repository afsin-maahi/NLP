# ============================================================
# TEXT PREPROCESSING FUNCTIONS
# ============================================================

import re
import html
import string
import emoji
import contractions
import nltk
from textblob import TextBlob
from pathlib import Path
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHAT_WORDS = {}
with open(DATA_DIR / "slang.txt", encoding="utf-8") as file:
    for line in file:
        key, value = line.strip().split("=", 1)
        CHAT_WORDS[key.lower()] = value.lower()



def remove_html(text):
    return BeautifulSoup(text, "html.parser").get_text()


def decode_html(text):
    return html.unescape(text)


def remove_urls(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_emojis(text):
    return emoji.replace_emoji(text, replace="")


def remove_extra_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


def lowercase(text):
    return text.lower()


def remove_special_characters(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text):
    return re.sub(r"\d+", "", text)


def expand_contractions(text):
    return contractions.fix(text)


def tokenize(text):
    return text.split()

stop_words = set(stopwords.words("english"))
def remove_stopwords(tokens):
    return [
        word
        for word in tokens
        if word not in stop_words
    ]


STEMMER = PorterStemmer()
def stemming(tokens):
    return [STEMMER.stem(word) for word in tokens]


LEMMATIZER = WordNetLemmatizer()
def lemmatization(tokens):
    return [LEMMATIZER.lemmatize(word) for word in tokens]


def remove_html_tags(text):
    pattern=re.compile('<.*?>')
    return pattern.sub(r'',text)


def expand_chat_words(text):
    words = text.split()
    expanded_words = []
    for word in words:
        expanded_words.append(
            CHAT_WORDS.get(word.lower(), word)
        )
    return " ".join(expanded_words)


def correct_spelling(text):
    return str(TextBlob(text).correct())


text = """
<html>
<body>

Hey!!! 😊

I'm learning NLP from CampusX.

Visit https://www.example.com

I've completed 100 exercises!!!

</body>
</html>
"""

text = remove_html(text)
text = decode_html(text)
text = remove_urls(text)
text = remove_emojis(text)
text = remove_extra_spaces(text)
text = lowercase(text)
text = remove_special_characters(text)
text = remove_numbers(text)
text = expand_contractions(text)
tokens = tokenize(text)
tokens = remove_stopwords(tokens)
tokens = lemmatization(tokens)
clean_text = " ".join(tokens)

# print(clean_text)

text = "I realy luv this movi"
print(correct_spelling(text))