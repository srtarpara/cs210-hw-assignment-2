# tfidf.py
import re
import math

# -------------------------------
# Part 1: Preprocessing
# -------------------------------

def clean_text(text):
    """Remove URLs, punctuation, and extra whitespace; lowercase all words."""
    # TODO: implement cleaning rules
    return text


def remove_stopwords(words, stopwords):
    """Remove stopwords from a list of words."""
    # TODO: filter out stopwords
    return words


def stem_word(word):
    """Apply basic stemming rules."""
    # TODO: reduce words ending with 'ing', 'ly', or 'ment'
    return word


def preprocess_file(filename, stopwords):
    """
    Read input file, clean, remove stopwords, apply stemming,
    and write output to 'preproc_<filename>'.
    """
    # TODO: open file, apply transformations, write output file
    pass


# -------------------------------
# Part 2: TF-IDF Computation
# -------------------------------

def compute_tf(words):
    """Compute term frequency for each word in a single document."""
    tf = {}
    # TODO: implement TF formula
    return tf


def compute_idf(all_docs):
    """
    Compute inverse document frequency across all documents.
    all_docs is a list of lists of words.
    """
    idf = {}
    # TODO: implement IDF formula
    return idf


def compute_tfidf(tf, idf):
    """Compute TF-IDF scores from TF and IDF dictionaries."""
    tfidf = {}
    # TODO: multiply TF * IDF and round to 2 decimals
    return tfidf


def write_tfidf_to_file(tfidf_scores, output_filename):
    """Write top 5 words and their TF-IDF scores to file."""
    # TODO: sort by score desc, then word asc
    pass


# -------------------------------
# Main driver
# -------------------------------

def main():
    # 1. Read list of documents from tfidf_docs.txt
    with open("tfidf_docs.txt", "r") as f:
        doc_names = [line.strip() for line in f if line.strip()]

    # 2. Load stopwords
    with open("stopwords.txt", "r") as f:
        stopwords = set(word.strip() for word in f)

    # 3. Preprocess each document
    for doc in doc_names:
        preprocess_file(doc, stopwords)

    # 4. Compute TF-IDF for each preprocessed document
    preprocessed_docs = []
    for doc in doc_names:
        with open(f"preproc_{doc}", "r") as f:
            words = f.read().split()
            preprocessed_docs.append(words)

    idf = compute_idf(preprocessed_docs)

    # 5. Compute and write TF-IDF for each document
    for i, doc in enumerate(doc_names):
        tf = compute_tf(preprocessed_docs[i])
        tfidf = compute_tfidf(tf, idf)
        write_tfidf_to_file(tfidf, f"tfidf_{doc}")

if __name__ == "__main__":
    main()
