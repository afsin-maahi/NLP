
import re
import html
import string
import unicodedata
from pathlib import Path
import emoji
import contractions
import nltk
from bs4 import BeautifulSoup
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()


CHAT_WORDS = {}
with open(DATA_DIR / "slang.txt", "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        CHAT_WORDS[key.lower()] = value.lower()


def remove_html(text):
    """
    Removes HTML tags from text.
    """
    return BeautifulSoup(text, "html.parser").get_text()


def decode_html(text):
    """
    Converts HTML entities into normal characters.
    """
    return html.unescape(text)


def remove_urls(text):
    """
    Removes URLs from text.
    """
    pattern = r"https?://\S+|www\.\S+"
    return re.sub(pattern, "", text)


def remove_emails(text):
    """
    Removes email addresses.
    """
    pattern = r"\S+@\S+"
    return re.sub(pattern, "", text)


def remove_phone_numbers(text):
    """
    Removes phone numbers.
    """
    pattern = r"\+?\d[\d\s()-]{7,}\d"
    return re.sub(pattern, "", text)


def remove_mentions(text):
    """
    Removes @username mentions.
    """
    return re.sub(r"@\w+", "", text)


def process_hashtags(text):
    """
    Removes '#' while keeping the hashtag word.
    """
    return re.sub(r"#(\w+)", r"\1", text)


def remove_emojis(text):
    """
    Removes emojis from text.
    """
    return emoji.replace_emoji(text, replace="")


def remove_extra_spaces(text):
    """
    Removes unnecessary whitespaces, tabs and newlines.
    """
    return re.sub(r"\s+", " ", text).strip()


def convert_to_lowercase(text):
    """
    Converts all characters to lowercase.
    """
    return text.lower()

# ============================================================
# Remove Accented Characters
# Example:
# café -> cafe
# résumé -> resume
# ============================================================

def remove_accents(text):
    """
    Removes accented characters.
    """
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("utf-8")

# ============================================================
#  Expand Chat Words / Slang
# Example:
# lol -> laughing out loud
# brb -> be right back
# ============================================================

def expand_chat_words(text):
    """
    Expands chat abbreviations using slang.txt.
    """

    def replace(match):
        word = match.group(0)
        return CHAT_WORDS.get(word.lower(), word)
    return re.sub(r"\b\w+\b", replace, text)

# ============================================================
#  Expand Contractions
# Example:
# can't -> cannot
# I'm -> I am
# ============================================================

def expand_contractions(text):
    """
    Expands English contractions.
    """
    return contractions.fix(text)


def correct_spelling(text):
    """
    Corrects spelling mistakes using TextBlob.
    (For learning purposes. Production systems often use
    SymSpell or transformer-based approaches.)
    """
    return str(TextBlob(text).correct())


def remove_special_characters(text):
    """
    Removes punctuation and special characters.
    """
    return text.translate(
        str.maketrans("", "", string.punctuation)
    )


def remove_numbers(text):
    """
    Removes all numeric characters.
    """
    return re.sub(r"\d+", "", text)


def tokenize_text(text):
    """
    Splits text into words.
    """
    return text.split()



def remove_stopwords(tokens):
    """
    Removes common English stopwords.
    """
    return [
        word
        for word in tokens
        if word not in STOP_WORDS
    ]



def apply_stemming(tokens):
    """
    Converts words to their root form.
    """
    return [
        STEMMER.stem(word)
        for word in tokens
    ]



def apply_lemmatization(tokens):
    """
    Converts words to their dictionary form.
    """
    return [
        LEMMATIZER.lemmatize(word)
        for word in tokens
    ]



def normalize_unicode(text):
    """
    Normalizes unicode characters.
    """
    return unicodedata.normalize("NFKC", text)


# ============================================================
#  Remove Repeated Characters
# Example:
# Soooooooo ---> Soo
# Happppppy ---> Happy
# ============================================================

def remove_repeated_characters(text):
    """
    Reduces repeated characters.
    """
    return re.sub(r"(.)\1{2,}", r"\1\1", text)



def remove_non_ascii(text):
    """
    Removes non ASCII characters.
    """
    return text.encode(
        "ascii",
        "ignore"
    ).decode(
        "ascii"
    )



def remove_empty_tokens(tokens):
    """
    Removes empty tokens.
    """
    return [
        token
        for token in tokens
        if token.strip()
    ]



def preprocess(
        text,
        remove_stop_words=True,
        remove_nums=False,
        apply_spell_check=False,
        apply_stem=False,
        apply_lemma=True
):

    text = remove_html(text)
    text = decode_html(text)
    text = normalize_unicode(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_phone_numbers(text)
    text = remove_mentions(text)
    text = process_hashtags(text)
    text = remove_emojis(text)
    text = remove_extra_spaces(text)
    text = convert_to_lowercase(text)
    text = remove_accents(text)
    text = expand_chat_words(text)
    text = expand_contractions(text)
    text = remove_repeated_characters(text)
    if apply_spell_check:
        text = correct_spelling(text)
    text = remove_special_characters(text)
    if remove_nums:
        text = remove_numbers(text)
    text = remove_non_ascii(text)
    tokens = tokenize_text(text)
    tokens = remove_empty_tokens(tokens)
    if remove_stop_words:
        tokens = remove_stopwords(tokens)
    if apply_stem:
        tokens = apply_stemming(tokens)
    if apply_lemma:
        tokens = apply_lemmatization(tokens)
    return " ".join(tokens)


if __name__ == "__main__":

    sample_text = """
    <html>

    Heyyyyy!!! 😂😂

    I'm learning NLP from CampusX.

    Visit https://www.example.com

    Email me at abc@gmail.com

    Call me at +91-9876543210

    idk lol this movi is gr8!!!!

    #MachineLearning

    </html>
    """
    clean_text = preprocess(
        sample_text,
        remove_stop_words=True,
        remove_nums=False,
        apply_spell_check=False,
        apply_stem=False,
        apply_lemma=True
    )
    print("=" * 60)
    print("FINAL PREPROCESSED TEXT")
    print("=" * 60)
    print(clean_text)