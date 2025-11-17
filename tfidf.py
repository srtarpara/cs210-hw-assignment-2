# tfidf.py
import re
import math

# -------------------------------
# Part 1: Preprocessing
# -------------------------------

def clean_text(text):
    """Remove URLs, punctuation, and extra whitespace; lowercase all words."""
    # Remove URLs (http:// or https://)
    text = re.sub(r'https?://\S+', '', text)

    # Keep only word characters and whitespace
    text = re.sub(r'[^\w\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_stopwords(words, stopwords):
    """Remove stopwords from a list of words."""
    return [word for word in words if word not in stopwords]


def stem_word(word):
    """Apply basic stemming rules."""
    if word.endswith('ing') and len(word) > 4:  # avoid words like 'string'
        return word[:-3]
    elif word.endswith('ly') and len(word) > 3:
        return word[:-2]
    elif word.endswith('ment') and len(word) > 4:
        return word[:-4]
    else:
        return word


def preprocess_file(filename, stopwords):
    """
    Read input file, clean, remove stopwords, apply stemming,
    and write output to 'preproc_<filename>'.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    # Clean
    text = clean_text(text)

    # Split into words
    words = text.split()

    # Remove stopwords
    words = remove_stopwords(words, stopwords)

    # Apply stemming
    words = [stem_word(word) for word in words]

    # Join words back into a single string
    processed_text = ' '.join(words)

    # Write to output file
    output_filename = f'preproc_{filename}'
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(processed_text)


# -------------------------------
# Part 2: TF-IDF Computation
# -------------------------------

def compute_tf(words):
    """Compute term frequency for each word in a single document."""
    tf = {}
    total_words = len(words)
    
    for word in words:
        tf[word] = tf.get(word, 0) + 1
    
    # Convert counts to frequencies
    for word in tf:
        tf[word] = tf[word] / total_words
    
    return tf


def compute_idf(all_docs):
    """
    Compute inverse document frequency across all documents.
    all_docs is a list of lists of words.
    """
    idf = {}
    N = len(all_docs)
    
    # Count in how many documents each word appears
    df = {}
    for doc in all_docs:
        unique_words = set(doc)
        for word in unique_words:
            df[word] = df.get(word, 0) + 1
    
    # Compute IDF
    for word, count in df.items():
        idf[word] = math.log(N / count) + 1
    
    return idf


def compute_tfidf(tf, idf):
    """Compute TF-IDF scores from TF and IDF dictionaries."""
    tfidf = {}
    for word in tf:
        tfidf[word] = round(tf[word] * idf.get(word, 0), 2)
    return tfidf


def write_tfidf_to_file(tfidf_scores, output_filename):
    """Write top 5 words and their TF-IDF scores to file."""
    # Sort: by score descending, then alphabetically
    sorted_words = sorted(tfidf_scores.items(), key=lambda x: (-x[1], x[0]))
    
    # Take top 5
    top5 = sorted_words[:5]
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(str(top5))


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