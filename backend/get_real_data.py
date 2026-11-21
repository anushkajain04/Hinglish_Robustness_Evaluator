"""
Expanded Dataset Ingestion & Noise-Cleaning Pipeline for Hinglish Sentiment Analysis
Downloads and merges multiple real-world Hinglish sentiment corpora:
  1. shae2977/hinglish-youtube-sentiments-dataset (YouTube comments)
  2. Abhishek4896/hindi-english-code-mixed-tweets-sentiment (Twitter Hinglish)
  3. RTT1/SentiMix (Official SemEval-2020 Task 9 SentiMix Hinglish corpus)
  4. Optional mirrors (rahul1990/hinglish-sentiment, l3cube-pune/hinglish-sentiment) in try-except blocks

Cleaning Rules:
- Strip URLs, @mentions, RT prefixes, and extra whitespace; remove '#' while keeping words.
- Drop sentences with fewer than 3 alphabetic words or containing only numbers/emojis/symbols.
- Normalize sentiment strictly to 'Positive', 'Negative', 'Neutral'.
- Remove conflicting labels (sentences mapped to different sentiments across annotations/sources).
- Deduplicate, shuffle, and save to backend/hinglish_data.csv.
- Removes stale baseline_pipeline.pkl / upgraded_pipeline.pkl to trigger automated model retraining.
"""

import os
import random
import re
import sys
import pandas as pd


def clean_hinglish_text(text: str) -> str:
    """
    Cleans raw text:
    - Strips URLs (http, https, www)
    - Strips @mentions
    - Strips 'RT' prefixes
    - Removes '#' symbols while preserving the hashtag word
    - Collapses multiple whitespace characters
    - Returns empty string if fewer than 3 alphabetic words exist
    """
    if not isinstance(text, str):
        return ""
    t = text
    # Remove URLs
    t = re.sub(r'https?://\S+|www\.\S+', '', t)
    # Remove user mentions
    t = re.sub(r'@[A-Za-z0-9_]+', '', t)
    # Remove RT prefix
    t = re.sub(r'^\s*rt[\s:]+', '', t, flags=re.IGNORECASE)
    # Remove '#' symbols while keeping hashtag words
    t = re.sub(r'#([A-Za-z0-9_]+)', r'\1', t)
    # Collapse multiple whitespaces
    t = re.sub(r'\s+', ' ', t).strip()

    # Drop text that contains fewer than 3 words with alphabetic characters
    alpha_words = [w for w in t.split() if re.search(r'[a-zA-Z]', w)]
    if len(alpha_words) < 3:
        return ""

    return t


def normalize_sentiment(val):
    """
    Strictly normalizes varied label representations to
    'Positive', 'Negative', or 'Neutral'.
    """
    if val is None or pd.isna(val):
        return None
    s = str(val).strip().lower()

    if "pos" in s or s in ["2", "2.0"]:
        return "Positive"
    if "neg" in s or s in ["0", "0.0", "-1"]:
        return "Negative"
    if "neu" in s or s in ["1", "1.0"]:
        return "Neutral"

    return None


def parse_sem_eval_sentimix(ds):
    """
    Parses RTT1/SentiMix (SemEval-2020 SentiMix Hinglish CONLL format)
    into sentence and label pairs.
    """
    sentences = []
    current_words = []
    current_label = None

    for row in ds:
        line = str(row.get('text', '')).strip()
        if line.startswith('meta\t'):
            if current_words and current_label:
                sentences.append((' '.join(current_words), current_label))
            parts = line.split('\t')
            current_label = parts[2] if len(parts) > 2 else None
            current_words = []
        elif '\t' in line:
            tok = line.split('\t')[0]
            current_words.append(tok)

    if current_words and current_label:
        sentences.append((' '.join(current_words), current_label))

    return pd.DataFrame(sentences, columns=["text", "label"])


def load_and_clean_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Loads a dataset from Hugging Face, applies schema detection and cleaning.
    """
    from datasets import load_dataset

    print(f"[*] Attempting to fetch '{dataset_name}' from Hugging Face...")
    ds = load_dataset(dataset_name)

    split_name = "train" if "train" in ds else list(ds.keys())[0]
    split_data = ds[split_name]
    print(f"    - Split '{split_name}' contains {len(split_data):,} records")

    # Special handling for SemEval SentiMix format
    if "SentiMix" in dataset_name or dataset_name == "RTT1/SentiMix":
        raw_df = parse_sem_eval_sentimix(split_data)
        print(f"    - Parsed {len(raw_df):,} full sentences from SentiMix stream")
        text_col, label_col = "text", "label"
    else:
        raw_df = split_data.to_pandas()

        # Dynamic column detection
        text_col = None
        for cand in ["text", "comment", "tweet", "sentence", "clean_text", "content"]:
            if cand in raw_df.columns:
                text_col = cand
                break

        label_col = None
        for cand in ["sentiment", "label", "category", "tag"]:
            if cand in raw_df.columns:
                label_col = cand
                break

        if not text_col or not label_col:
            raise ValueError(f"Could not resolve text/label columns from: {list(raw_df.columns)}")

    df = raw_df[[text_col, label_col]].copy()
    df = df.rename(columns={text_col: "text", label_col: "label"})

    # Normalize labels
    df["label"] = df["label"].apply(normalize_sentiment)
    df = df.dropna(subset=["label"])
    df = df[df["label"].isin(["Positive", "Negative", "Neutral"])].copy()

    # Pre-clean text strings
    df["text"] = df["text"].astype(str).apply(clean_hinglish_text)
    df = df[df["text"] != ""].copy()

    return df[["text", "label"]]


def generate_offline_hinglish_dataset(n_samples=500):
    """
    Programmatic fallback generator in case all external downloads fail.
    """
    subjects = [
        "bhai ye video", "bhai ye phone", "ye movie", "khana", "service",
        "camera quality", "acting", "ye song", "bhai content", "product",
        "bhai ye app", "screen display", "battery backup", "delivery", "ye scene"
    ]
    pos_adjectives = ["bohot mast", "ekdum zabardast", "kamaal ka", "bahut achha", "full paisa vasool", "next level"]
    neg_adjectives = ["ekdum bakwas", "bohot ghatiya", "bilkul bekar", "worst ever", "paisa barbad"]
    neu_adjectives = ["theek thak", "average", "normal sa", "decent", "bas kaam chalau", "ok ok"]

    data = []
    counts = {"Positive": n_samples // 3, "Negative": n_samples // 3, "Neutral": n_samples - 2 * (n_samples // 3)}
    for label, target_count in counts.items():
        adjs = pos_adjectives if label == "Positive" else (neg_adjectives if label == "Negative" else neu_adjectives)
        for i in range(target_count):
            t = f"{random.choice(subjects)} {random.choice(adjs)} laga mujhe"
            data.append({"text": t, "label": label})

    random.shuffle(data)
    return pd.DataFrame(data)


def fetch_and_combine_datasets():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "hinglish_data.csv")

    candidate_sources = [
        "shae2977/hinglish-youtube-sentiments-dataset",
        "Abhishek4896/hindi-english-code-mixed-tweets-sentiment",
        "RTT1/SentiMix",
        "rahul1990/hinglish-sentiment",
        "l3cube-pune/hinglish-sentiment"
    ]

    loaded_dfs = []
    source_stats = {}

    for name in candidate_sources:
        try:
            df_cleaned = load_and_clean_dataset(name)
            loaded_dfs.append(df_cleaned)
            source_stats[name] = len(df_cleaned)
            print(f"[+] Successfully integrated '{name}' ({len(df_cleaned):,} valid records)")
        except Exception as err:
            print(f"[-] Skipped '{name}': {err}")

    if loaded_dfs:
        print("\n[*] Merging all dataset partitions...")
        combined_df = pd.concat(loaded_dfs, ignore_index=True)
        raw_total = len(combined_df)
        print(f"[*] Total merged records: {raw_total:,}")

        # Aggressive Cleaning:
        # 1. Clean and filter text length
        combined_df["text"] = combined_df["text"].astype(str).apply(clean_hinglish_text)
        len_filtered_count = len(combined_df)
        combined_df = combined_df[combined_df["text"] != ""].copy()
        short_dropped = len_filtered_count - len(combined_df)

        # 2. Conflict Removal:
        # Group by lowercased, stripped text. If the same text has multiple different sentiment labels,
        # it is noisy annotation — drop ALL conflicting instances!
        combined_df["clean_key"] = combined_df["text"].str.lower().str.strip()
        label_nunique = combined_df.groupby("clean_key")["label"].transform("nunique")
        conflicts_count = (label_nunique > 1).sum()
        print(f"[*] Detected {conflicts_count:,} conflicting label instances across datasets. Removing all conflicting rows...")
        combined_df = combined_df[label_nunique == 1].copy()

        # 3. Deduplication:
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=["clean_key"]).drop(columns=["clean_key"]).reset_index(drop=True)
        duplicates_dropped = before_dedup - len(combined_df)

        # 4. Shuffle rows
        combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        total_noise_removed = raw_total - len(combined_df)
        source_summary = f"Multi-Source Hugging Face Corpus ({len(loaded_dfs)} sources)"
    else:
        print("\n[!] All Hugging Face dataset downloads failed.")
        print("[*] Engaging last-resort Offline Synthetic Generator...")
        combined_df = generate_offline_hinglish_dataset(n_samples=1000)
        source_summary = "Offline Synthetic Generator (Fallback)"
        source_stats["OfflineGenerator"] = len(combined_df)
        total_noise_removed = 0

    # Save to CSV
    combined_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n[+] Saved cleaned corpus to: {output_path}")

    # Remove stale baseline and upgraded pipeline artifacts
    for artifact in ["baseline_pipeline.pkl", "upgraded_pipeline.pkl", "model.pkl", "vectorizer.pkl"]:
        artifact_path = os.path.join(script_dir, artifact)
        if os.path.exists(artifact_path):
            try:
                os.remove(artifact_path)
                print(f"[-] Removed stale artifact: '{artifact}'")
            except Exception as e:
                print(f"[!] Warning removing {artifact}: {e}")

    # Print summary metrics
    print("\n" + "=" * 70)
    print("DATA PIPELINE EXPANSION & CLEANING SUMMARY")
    print("=" * 70)
    print(f"Pipeline Source          : {source_summary}")
    print(f"Output File              : {output_path}")
    print("\nContribution by Source   :")
    for src, count in source_stats.items():
        print(f"  - {src:<55}: {count:,} records")
    print(f"\nTotal Merged Raw Rows    : {raw_total if loaded_dfs else len(combined_df):,}")
    print(f"Total Noise Rows Removed : {total_noise_removed:,}")
    print(f"Final Cleaned Rows Saved : {len(combined_df):,}")
    print("\nFinal Class Distribution :")
    for label, count in combined_df["label"].value_counts().items():
        pct = (count / len(combined_df)) * 100
        print(f"  - {label:<10}: {count:,} ({pct:.1f}%)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    fetch_and_combine_datasets()
