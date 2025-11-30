
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
import numpy as np

nltk.download('stopwords', quiet=True)

# Here, I am defining the set of common English stop words for efficient lookup.
STOP = set(stopwords.words('english'))

def simple_tokenize(text):
    """
    Here, I am performing a basic text cleaning and tokenization.
    I am converting the text to lowercase, removing most punctuation,
    and then splitting it into tokens (words). Finally, I am filtering out
    single-character tokens and common English stop words.
    """
    # Convert input to string and lowercase it.
    text = str(text).lower()
    # Remove all characters that are not a-z, 0-9, or whitespace.
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Split the text into tokens (words).
    # I am keeping tokens that are longer than 1 character AND are not stop words.
    toks = [t for t in text.split() if len(t) > 1 and t not in STOP]
    return toks

def preprocess_file(csv_path, text_col='plot', max_docs=None):
    """
    Here, I am reading the raw data, applying constraints, and orchestrating the cleaning process.
    I am loading the CSV file into a pandas DataFrame.
    I am enforcing a 'max_docs' limit if provided, which is useful for rapid prototyping.
    Finally, I am applying the 'simple_tokenize' function to the specified text column
    to get lists of clean tokens and their joined string versions.
    """
    # Load the CSV data.
    df = pd.read_csv(csv_path)

    # I am doing a basic check to make sure the specified text column exists.
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Available: {df.columns}")

    # Limit the number of documents if 'max_docs' is specified for quick runs.
    if max_docs:
        df = df.iloc[:max_docs]

    # Extract the text column, ensuring all NaNs are replaced with empty strings, and convert to a list.
    raw_texts = df[text_col].fillna("").astype(str).tolist()
    
    # Apply tokenization to every document.
    token_lists = [simple_tokenize(t) for t in raw_texts]
    
    # Create the joined string version of the tokens (needed for TF-IDF/LSI).
    joined = [" ".join(toks) for toks in token_lists]
    
    # I am returning the tokenized data and the original filtered DataFrame.
    return token_lists, joined, df

if __name__ == "__main__":
    # Here, I am setting up the command-line interface (CLI) so this script is runnable directly.
    import argparse, os
    parser = argparse.ArgumentParser()
    # I am requiring the path to the input CSV.
    parser.add_argument("--csv", required=True, help="path to CSV")
    # I am requiring the output path for the processed data (as an NPZ file).
    parser.add_argument("--out", required=True, help="output npz path")
    # I am allowing the user to specify which column holds the text (default is 'plot').
    parser.add_argument("--text_col", default="plot", help="column name with text")
    # I am allowing an optional limit on the number of documents to process.
    parser.add_argument("--max_docs", type=int, default=None)
    args = parser.parse_args()

    # I am calling the main preprocessing function.
    toks, joined, df = preprocess_file(args.csv, text_col=args.text_col, max_docs=args.max_docs)
    
    # I am ensuring the output directory exists before saving the file.
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Here, I am saving the processed data (token lists and joined strings) into a compressed NPZ file.
    # The 'dtype=object' is important for saving ragged arrays (lists of varying length) correctly.
    np.savez_compressed(
        args.out,
        token_lists=np.array(toks, dtype=object),
        joined=np.array(joined, dtype=object),
    )

    print("Saved processed data to", args.out)




