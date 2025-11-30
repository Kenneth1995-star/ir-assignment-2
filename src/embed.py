import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
from datasketch import MinHash
import joblib

def embed_lsi(joined_strings, n_components=200, tfidf_max_features=50000):
    """
    Here, I am generating Latent Semantic Indexing (LSI) embeddings.
    I am first using TF-IDF to convert the joined text strings into a sparse matrix, 
    limiting the vocabulary size to `tfidf_max_features`.
    Then, I am applying Truncated SVD to reduce the dimensionality from the large
    vocabulary space down to `n_components` (e.g., 200), which creates the dense LSI vectors.
    """
    # I am initializing the TF-IDF vectorizer and fitting/transforming the data.
    vect = TfidfVectorizer(max_features=tfidf_max_features)
    Xtf = vect.fit_transform(joined_strings)
    
    # I am initializing Truncated SVD, which is used for LSI.
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    # I am applying SVD to the TF-IDF matrix to get the dense LSI embeddings.
    X_lsi = svd.fit_transform(Xtf)
    
    # I am returning the embeddings and the fitted models for later use in querying.
    return X_lsi, vect, svd

def embed_word2vec(token_lists, size=200, window=5, min_count=2, sg=1, epochs=5):
    """
    Here, I am generating Word2Vec (W2V) document embeddings.
    First, I am training a Skip-gram (sg=1) Word2Vec model on the tokenized text.
    Then, I am creating a document vector by simply averaging the word vectors
    for all words present in the model's vocabulary within that document.
    """
    # I am training the Word2Vec model on the lists of tokens.
    model = Word2Vec(sentences=token_lists, vector_size=size, window=window, min_count=min_count, sg=sg, epochs=epochs)
    
    # This helper function calculates the vector for a single document (list of tokens).
    def doc_vector(tokens):
        # I am extracting vectors for all tokens that the model knows.
        vecs = [model.wv[t] for t in tokens if t in model.wv]
        # If no words were found in the vocabulary, I am returning a zero vector.
        if len(vecs) == 0:
            return np.zeros(size, dtype=float)
        # Otherwise, I am returning the average of the word vectors.
        return np.mean(vecs, axis=0)
        
    # I am applying the doc_vector function to all token lists and stacking them into a matrix.
    X = np.vstack([doc_vector(toks) for toks in token_lists])
    
    # I am returning the W2V embeddings and the trained model.
    return X, model

# ---------- SHINGLES for MinHash (fix for LSH recall) ----------
def make_shingles(tokens, n=5):
    """
    Here, I am creating word shingles (n-grams) from the list of tokens.
    Shingles are contiguous sequences of words, which are essential for MinHash
    as they capture local phrase context and are used to calculate Jaccard similarity.
    """
    if not tokens:
        return []
    # If the document is too short, I am just using the whole document as one shingle.
    if len(tokens) < n:
        return [" ".join(tokens)]
    # I am generating all overlapping n-word sequences.
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def minhash_sketch(token_lists, num_perm=128, shingle_n=5):
    """
    Here, I am generating a MinHash sketch for every document.
    This sketch is a compact representation that allows for fast estimation of
    Jaccard similarity between documents, which is necessary for LSH indexing.
    """
    sketches = []
    for toks in token_lists:
        # I am creating a new MinHash object for the document.
        mh = MinHash(num_perm=num_perm)
        # I am converting the tokens into shingles (e.g., 5-word phrases).
        shingles = make_shingles(toks, n=shingle_n)
        # I am updating the MinHash with each shingle, encoding them to bytes first.
        for sh in shingles:
            mh.update(sh.encode('utf8'))
        sketches.append(mh)
        
    # I am returning the list of MinHash objects (sketches).
    return sketches

if __name__ == "__main__":
    # Here, I am setting up the command-line interface to select and run one of the three embedding methods.
    import argparse, numpy as np, os
    parser = argparse.ArgumentParser()
    # Input is the NPZ file saved by preprocess.py.
    parser.add_argument("--processed_npz", required=True)
    # I am allowing the user to choose LSI, W2V, or MinHash.
    parser.add_argument("--method", choices=["lsi","w2v","minhash"], default="lsi")
    # Output path for the embeddings or sketches.
    parser.add_argument("--out", required=True)
    # Common argument for dimension size (LSI/W2V).
    parser.add_argument("--n_components", type=int, default=200)
    # Arguments specific to MinHash.
    parser.add_argument("--num_perm", type=int, default=128)
    parser.add_argument("--shingle_n", type=int, default=5)
    args = parser.parse_args()

    # I am loading the preprocessed data: lists of tokens and joined strings.
    data = np.load(args.processed_npz, allow_pickle=True)
    token_lists = list(data["token_lists"])
    joined = list(data["joined"])

    # I am executing the chosen embedding method.
    if args.method == "lsi":
        X, vect, svd = embed_lsi(joined, n_components=args.n_components)
        # I am saving the LSI embeddings as a NumPy array.
        np.save(args.out, X)
        print("Saved LSI embeddings to", args.out)
    elif args.method == "w2v":
        X, model = embed_word2vec(token_lists, size=args.n_components)
        # I am saving the W2V embeddings as a NumPy array.
        np.save(args.out, X)
        # I am also saving the trained Word2Vec model separately.
        model.save(args.out + ".w2v")
        print("Saved W2V embeddings + model to", args.out)
    else: # MinHash
        sketches = minhash_sketch(token_lists, num_perm=args.num_perm, shingle_n=args.shingle_n)
        # I am using joblib to save the list of MinHash objects.
        joblib.dump(sketches, args.out)
        print("Saved MinHash sketches (shingles) to", args.out)





