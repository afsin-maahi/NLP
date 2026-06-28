
import re
import nltk
from nltk.tokenize import (
    sent_tokenize,
    word_tokenize,
    RegexpTokenizer,
    TreebankWordTokenizer,
    TweetTokenizer,
    MWETokenizer
)
nltk.download("punkt")
nltk.download("punkt_tab")
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer
import sentencepiece as spm
from transformers import AutoTokenizer



sample_text = """
Hello everyone!
I'm learning Natural Language Processing.
NLP is one of the most exciting fields in AI.
Visit https://openai.com to learn more.
My email is abc@gmail.com.
Today's date is 28/06/2026.
I scored 95% in NLP!!
#MachineLearning
"""

tweet_text = """
OMG!!!! 😂😂
I'm loving #MachineLearning
Follow @OpenAI
Visit https://openai.com
NLP is soooo coooool!!!!
"""

mwe_text = """
Machine learning and Natural Language Processing
are exciting areas of Artificial Intelligence.
"""

training_corpus = [
    "Natural Language Processing is amazing.",
    "Machine Learning is fun.",
    "Deep Learning powers modern AI.",
    "Artificial Intelligence is transforming the world.",
    "I love learning NLP.",
    "ChatGPT is based on Transformers."
]

def sentence_tokenization(text):
    """
    Splits text into sentences.
    Example:
    Hello. How are you?
    Output:
    ['Hello.', 'How are you?']
    """
    return sent_tokenize(text)


def word_tokenization(text):
    """
    Splits text into words and punctuation.
    Example:
    I love NLP!
    Output:
    ['I', 'love', 'NLP', '!']
    """
    return word_tokenize(text)


def character_tokenization(text):
    """
    Splits text into individual characters.
    Example:
    NLP
    Output:
    ['N', 'L', 'P']
    """
    return list(text)


def regex_tokenization(text):
    """
    Tokenizes text using a custom regular expression.
    This tokenizer extracts only words.
    Example:
    NLP, AI!!
    Output:
    ['NLP', 'AI']
    """
    tokenizer = RegexpTokenizer(r"\w+")
    return tokenizer.tokenize(text)


def treebank_tokenization(text):
    """
    Penn Treebank Tokenizer.
    Splits contractions intelligently.
    Example:
    I'm learning NLP.
    Output:
    ['I', "'m", 'learning', 'NLP', '.']
    """
    tokenizer = TreebankWordTokenizer()
    return tokenizer.tokenize(text)


def tweet_tokenization(text):
    """
    Tokenizer designed for social media text.
    Preserves:
    - Emojis
    - Hashtags
    - Mentions
    - Repeated letters
    """
    tokenizer = TweetTokenizer(
        preserve_case=False,
        strip_handles=False,
        reduce_len=True
    )
    return tokenizer.tokenize(text)

def mwe_tokenization(text):
    """
    Keeps predefined phrases as one token.
    Example:
    machine learning
    becomes
    machine_learning
    """
    tokenizer = MWETokenizer()
    tokenizer.add_mwe(("machine", "learning"))
    tokenizer.add_mwe(("natural", "language"))
    tokenizer.add_mwe(("artificial", "intelligence"))
    words = word_tokenize(text.lower())
    return tokenizer.tokenize(words)


def custom_tokenization(text):
    """
    Simple tokenizer built from scratch.
    Keeps only alphabetic words.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = text.split()
    return tokens


def train_bpe_tokenizer(corpus):
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(
        vocab_size=100,
        special_tokens=["[UNK]", "[PAD]"]
    )
    tokenizer.train_from_iterator(corpus, trainer)
    return tokenizer


def bpe_tokenization(text):
    tokenizer = train_bpe_tokenizer(training_corpus)
    encoding = tokenizer.encode(text)
    return encoding.tokens


def wordpiece_tokenization(text):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer.tokenize(text)


def train_sentencepiece():
    with open("sentencepiece_corpus.txt", "w", encoding="utf-8") as file:
        for sentence in training_corpus:
            file.write(sentence + "\n")
    spm.SentencePieceTrainer.train(
        input="sentencepiece_corpus.txt",
        model_prefix="sp",
        vocab_size=100
    )


def sentencepiece_tokenization(text):
    train_sentencepiece()
    sp = spm.SentencePieceProcessor()
    sp.load("sp.model")
    return sp.encode(text, out_type=str)


def huggingface_tokenization(text):
    """
    Returns all important tokenizer outputs.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased"
    )
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=20,
        return_tensors=None
    )
    return {
        "Tokens":
            tokenizer.convert_ids_to_tokens(
                encoded["input_ids"]
            ),
        "Token IDs":
            encoded["input_ids"],
        "Attention Mask":
            encoded["attention_mask"],
        "Decoded":
            tokenizer.decode(
                encoded["input_ids"],
                skip_special_tokens=True
            )
    }


def compare_transformer_tokenizers(text):
    bert = AutoTokenizer.from_pretrained(
        "bert-base-uncased"
    )
    gpt = AutoTokenizer.from_pretrained(
        "gpt2"
    )
    # GPT does not have a padding token by default
    gpt.pad_token = gpt.eos_token
    return {
        "BERT Tokens":
            bert.tokenize(text),
        "BERT IDs":
            bert.encode(text),
        "GPT Tokens":
            gpt.tokenize(text),
        "GPT IDs":
            gpt.encode(text)
    }


def build_vocabulary(corpus):
    vocabulary = {}
    index = 0
    for sentence in corpus:
        words = word_tokenize(sentence.lower())
        for word in words:
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1
    return vocabulary



def encode_sentence(sentence, vocabulary):
    encoded = []
    for word in word_tokenize(sentence.lower()):
        encoded.append(
            vocabulary.get(word, -1)
        )
    return encoded


def decode_sentence(ids, vocabulary):
    reverse_vocab = {
        value: key
        for key, value in vocabulary.items()
    }
    words = []
    for i in ids:
        words.append(
            reverse_vocab.get(i, "[UNK]")
        )
    return " ".join(words)


def print_tokens(title, tokens):
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(tokens)
    print()


if __name__ == "__main__":
    print_tokens(
        "Sentence Tokenization",
        sentence_tokenization(sample_text)
    )
    print_tokens(
        "Word Tokenization",
        word_tokenization(sample_text)
    )
    print_tokens(
        "Character Tokenization",
        character_tokenization("ChatGPT")
    )
    print_tokens(
        "Regex Tokenization",
        regex_tokenization(sample_text)
    )
    print_tokens(
        "Treebank Tokenizer",
        treebank_tokenization(sample_text)
    )
    print_tokens(
        "Tweet Tokenizer",
        tweet_tokenization(tweet_text)
    )
    print_tokens(
        "Multi Word Expression Tokenizer",
        mwe_tokenization(mwe_text)
    )
    print_tokens(
        "Custom Tokenizer",
        custom_tokenization(sample_text)
    )
    print_tokens(
        "Byte Pair Encoding",
        bpe_tokenization("unhappiness")
    )
    print_tokens(
        "WordPiece",
        wordpiece_tokenization("unhappiness")
    )
    print_tokens(
        "SentencePiece",
        sentencepiece_tokenization("unhappiness")
    )
    
    text = "I love studying Natural Language Processing."

    print("=" * 60)
    print("HUGGING FACE TOKENIZER")
    print("=" * 60)
    output = huggingface_tokenization(text)
    for key, value in output.items():
        print(f"\n{key}")
        print(value)
    print()
    print("=" * 60)
    print("BERT vs GPT")
    print("=" * 60)
    comparison = compare_transformer_tokenizers(text)
    for key, value in comparison.items():
        print(f"\n{key}")
        print(value)
    print()
    print("=" * 60)
    print("CUSTOM VOCABULARY")
    print("=" * 60)
    vocabulary = build_vocabulary(training_corpus)
    print(vocabulary)
    print()
    encoded = encode_sentence(
        "I love learning NLP",
        vocabulary
    )
    print("Encoded")
    print(encoded)
    print()
    decoded = decode_sentence(
        encoded,
        vocabulary
    )
    print("Decoded")
    print(decoded)