## Command to run the program
# python tfidf.py --input "/Users/jeromekithinji/Desktop/Concordia /Study/Fall 2025/COMP 6321 - ML/Project/COMP6321-SentimentAnalysis/data/Cell_Phones_and_Accessories_5.json" --outdir . --ngrams 1,2 --max_features 50000 --min_df 2 --max_df 0.95


# Load the data
# import kagglehub
# path = kagglehub.dataset_download("abdallahwagih/amazon-reviews")
# print("Path to dataset files:", path)

import argparse, os, json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
from joblib import dump

def load_reviews(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".json", ".jsonl"]:
        # JSON Lines (one review per line)
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return pd.DataFrame(rows)
    elif ext in [".csv", ".tsv"]:
        sep = "," if ext == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def build_text_column(df: pd.DataFrame) -> pd.Series:
    # Combine summary + reviewText
    summary = df.get("summary", "").fillna("")
    body = df.get("reviewText", "").fillna("")
    text = (summary + " " + body).str.strip()
    # If summary + reviewText, try "text" column
    if (text == "").all() and "text" in df.columns:
        text = df["text"].fillna("").astype(str)
    
    print(df.columns)
    print(df["reviewText"].head())
    print(df["summary"].head())

    return text



def main():
    ap = argparse.ArgumentParser(description="Fit TF-IDF on Amazon reviews and save artifacts.")
    ap.add_argument("--input", required=True, help="Path to reviews file (JSONL/JSON/CSV/TSV).")
    ap.add_argument("--outdir", default="artifacts", help="Where to save vectorizer & matrices.")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test split fraction.")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--max_features", type=int, default=50000)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--max_df", type=float, default=1.0)
    ap.add_argument("--ngrams", type=str, default="1,2", help="n-gram range like '1,2' or '1,3'.")
    ap.add_argument("--stop_words", default="english", help="Stopword setting (e.g., 'english' or None).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading data…")
    df = load_reviews(args.input)
    texts = build_text_column(df)
    print("Test Text head: ",texts.head(5))


    X_train_text, X_test_text = train_test_split(
        texts, test_size=args.test_size, random_state=args.random_state
    )

    ngram_range = tuple(int(x) for x in args.ngrams.split(","))
    print(f"Fitting TF-IDF (max_features={args.max_features}, ngram_range={ngram_range}, "
          f"min_df={args.min_df}, max_df={args.max_df}, stop_words={args.stop_words})…")

    # tfidf = TfidfVectorizer(
    #     lowercase=True,
    #     stop_words=None if args.stop_words.lower() == "none" else args.stop_words,
    #     ngram_range=ngram_range,
    #     max_features=args.max_features,
    #     min_df=args.min_df,
    #     max_df=args.max_df,
    # )

    tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",  # only alphabetic tokens (min 2 chars)
    ngram_range=(1, 2),
    max_features=50000,
    min_df=2,
    max_df=0.95
  )


    Xtr = tfidf.fit_transform(X_train_text)
    Xte = tfidf.transform(X_test_text)

    # Save artifacts
    vec_path = os.path.join(args.outdir, "vectorizer.joblib")
    Xtr_path = os.path.join(args.outdir, "X_train_tfidf.npz")
    Xte_path = os.path.join(args.outdir, "X_test_tfidf.npz")
    dump(tfidf, vec_path)
    save_npz(Xtr_path, Xtr)
    save_npz(Xte_path, Xte)

    print("Done")
    print(f"Vectorizer: {vec_path}")
    print(f"Train matrix: {Xtr_path}  shape={Xtr.shape}")
    print(f"Test  matrix: {Xte_path}  shape={Xte.shape}")

    # Check vocab
    vocab = tfidf.get_feature_names_out()
    print("Vocab size:", len(tfidf.get_feature_names_out()))
    print("Sample vocab:", ", ".join(vocab[:50]))

if __name__ == "__main__":
    main()
