import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from gensim.models import Word2Vec
from gensim.models import FastText
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


corpus = [
    ["i", "love", "machine", "learning"],
    ["machine", "learning", "is", "fun"],
    ["nlp", "is", "awesome"],
    ["deep", "learning", "uses", "embeddings"],
    ["artificial", "intelligence", "is", "the", "future"]
]

def build_cooccurrence_matrix(corpus, window_size=2):
    """
    Builds a co-occurrence matrix from the corpus.
    """
    vocabulary = {}
    index = 0
    for sentence in corpus:
        for word in sentence:
            if word not in vocabulary:
                vocabulary[word] = index
                index += 1
    matrix = np.zeros(
        (
            len(vocabulary),
            len(vocabulary)
        ),
        dtype=int
    )
    for sentence in corpus:
        for i, word in enumerate(sentence):
            word_index = vocabulary[word]
            start = max(0, i - window_size)
            end = min(len(sentence), i + window_size + 1)
            for j in range(start, end):
                if i == j:
                    continue
                neighbor = sentence[j]
                neighbor_index = vocabulary[neighbor]
                matrix[word_index][neighbor_index] += 1
    return vocabulary, matrix

def train_cbow(
        corpus,
        vector_size=100,
        window=2,
        epochs=100
):
    """
    Trains a CBOW Word2Vec model.
    """
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=1,
        sg=0,
        workers=4,
        epochs=epochs
    )
    return model


def train_skipgram(
        corpus,
        vector_size=100,
        window=2,
        epochs=100
):
    """
    Trains a Skip-Gram Word2Vec model.
    """
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=1,
        sg=1,
        workers=4,
        epochs=epochs
    )
    return model


def load_pretrained_word2vec():
    """
    Loads Google's pretrained Word2Vec model.
    """
    model = api.load("word2vec-google-news-300")
    return model


def find_similar_words(model, word, topn=10):
    """
    Finds the most similar words.
    """
    if word not in model.wv:
        return f"{word} not found in vocabulary."
    return model.wv.most_similar(word, topn=topn)


def word_analogy(model, positive, negative, topn=5):
    """
    Solves word analogies.
    Example:
    king - man + woman = queen
    """
    return model.wv.most_similar(
        positive=positive,
        negative=negative,
        topn=topn
    )


def calculate_cosine_similarity(model, word1, word2):
    """
    Calculates cosine similarity between two words.
    """
    if word1 not in model.wv:
        return None
    if word2 not in model.wv:
        return None
    vector1 = model.wv[word1].reshape(1, -1)
    vector2 = model.wv[word2].reshape(1, -1)
    similarity = cosine_similarity(
        vector1,
        vector2
    )
    return similarity[0][0]


def sentence_embedding(model, sentence):
    """
    Creates a sentence embedding by averaging word vectors.
    """
    vectors = []
    for word in sentence.split():
        if word in model.wv:
            vectors.append(
                model.wv[word]
            )
    if len(vectors) == 0:
        return None
    return np.mean(
        vectors,
        axis=0
    )


def train_fasttext(
        corpus,
        vector_size=100,
        window=2,
        epochs=100
):
    """
    Trains a FastText model.
    """
    model = FastText(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=4,
        epochs=epochs
    )
    return model


def load_pretrained_fasttext():
    """
    Loads Facebook FastText vectors.
    """
    model = api.load("fasttext-wiki-news-subwords-300")
    return model


def compare_word2vec_fasttext(word2vec_model, fasttext_model, word):
    """
    Compares Word2Vec and FastText embeddings.
    """
    print("=" * 60)
    print("WORD2VEC vs FASTTEXT")
    print("=" * 60)
    if word in word2vec_model.wv:
        print("\nWord2Vec Similar Words\n")
        print(
            word2vec_model.wv.most_similar(word)
        )
    else:
        print(f"{word} not found in Word2Vec vocabulary.")
    print()
    print("-" * 60)
    print()
    print("FastText Similar Words\n")
    print(
        fasttext_model.wv.most_similar(word)
    )


def visualize_embeddings_pca(model):
    """
    Visualizes word embeddings using PCA.
    """
    words = list(model.wv.index_to_key)
    vectors = model.wv[words]
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1])
    for i, word in enumerate(words):
        plt.annotate(
            word,
            (reduced[i, 0], reduced[i, 1])
        )
    plt.title("Word Embeddings using PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


def visualize_embeddings_tsne(model):
    """
    Visualizes word embeddings using t-SNE.
    """
    words = list(model.wv.index_to_key)
    vectors = model.wv[words]
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(5, len(words)-1)
    )
    reduced = tsne.fit_transform(vectors)
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1])
    for i, word in enumerate(words):
        plt.annotate(
            word,
            (reduced[i, 0], reduced[i, 1])
        )
    plt.title("Word Embeddings using t-SNE")
    plt.show()


def compare_embedding_methods():
    """
    Displays comparison of embedding techniques.
    """
    comparison = pd.DataFrame({
        "Method": [
            "One Hot Encoding",
            "Bag of Words",
            "TF-IDF",
            "Word2Vec",
            "FastText"
        ],

        "Captures Meaning": [
            "No",
            "No",
            "Partially",
            "Yes",
            "Yes"
        ],

        "Handles Unknown Words": [
            "No",
            "No",
            "No",
            "No",
            "Yes"
        ],

        "Word Order": [
            "No",
            "No",
            "No",
            "Context Window",
            "Context Window"
        ],

        "Embedding Size": [
            "Vocabulary Size",
            "Vocabulary Size",
            "Vocabulary Size",
            "Fixed",
            "Fixed"
        ]
    })
    print(comparison)



if __name__ == "__main__":

    print("=" * 60)
    print("CO-OCCURRENCE MATRIX")
    print("=" * 60)

    vocabulary, matrix = build_cooccurrence_matrix(corpus)

    print(vocabulary)
    print()
    print(matrix)
    print()


    print("=" * 60)
    print("CBOW MODEL")
    print("=" * 60)

    cbow_model = train_cbow(corpus)

    print(cbow_model.wv["learning"])
    print()


    print("=" * 60)
    print("SKIP-GRAM MODEL")
    print("=" * 60)

    skip_model = train_skipgram(corpus)

    print(skip_model.wv["learning"])
    print()


    print("=" * 60)
    print("SIMILAR WORDS")
    print("=" * 60)

    print(
        find_similar_words(
            cbow_model,
            "learning"
        )
    )

    print()


    print("=" * 60)
    print("WORD ANALOGY")
    print("=" * 60)

    try:

        print(
            word_analogy(
                cbow_model,
                positive=["learning"],
                negative=["machine"]
            )
        )

    except Exception as e:

        print(e)

    print()


    print("=" * 60)
    print("COSINE SIMILARITY")
    print("=" * 60)

    print(

        calculate_cosine_similarity(

            cbow_model,

            "machine",

            "learning"

        )

    )

    print()


    print("=" * 60)
    print("SENTENCE EMBEDDING")
    print("=" * 60)

    embedding = sentence_embedding(

        cbow_model,

        "machine learning is fun"

    )

    print(embedding)

    print()

    print("Embedding Dimension :", len(embedding))

    print()


    print("=" * 60)
    print("FASTTEXT MODEL")
    print("=" * 60)

    fasttext_model = train_fasttext(corpus)

    print(

        fasttext_model.wv["learning"]

    )

    print()


    print("=" * 60)
    print("COMPARE WORD2VEC AND FASTTEXT")
    print("=" * 60)

    compare_word2vec_fasttext(

        cbow_model,

        fasttext_model,

        "learning"

    )

    print()


    print("=" * 60)
    print("EMBEDDING METHODS")
    print("=" * 60)

    compare_embedding_methods()

    print()


    print("=" * 60)
    print("PCA VISUALIZATION")
    print("=" * 60)

    visualize_embeddings_pca(cbow_model)

    print()


    print("=" * 60)
    print("TSNE VISUALIZATION")
    print("=" * 60)

    visualize_embeddings_tsne(cbow_model)