import os
import random
import re
import sys
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

# Ensure backend directory is in sys.path for robust Joblib serialization/deserialization
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from nlp_utils import (
    HINGLISH_CANONICAL_MAP,
    LEXICAL_PERTURBATIONS,
    HinglishNegationTagger,
    HinglishNormalizer,
    normalize_hinglish,
    tag_hinglish_negation,
)

app = FastAPI(title="Hinglish Robustness Suite Backend", version="3.0.0")

# Allow React frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File paths
BASELINE_PATH = os.path.join(BACKEND_DIR, "baseline_pipeline.pkl")
UPGRADED_PATH = os.path.join(BACKEND_DIR, "upgraded_pipeline.pkl")
DATA_PATH = os.path.join(BACKEND_DIR, "hinglish_data.csv")

# Global pipeline instances and evaluation metrics
baseline_pipeline = None
upgraded_pipeline = None
MODEL_METRICS = {}


def clean_corpus_text(text: str) -> str:
    """
    Cleans raw text before train/test splitting:
    - Strips leading and trailing whitespace
    - Removes URLs (http, https, www)
    - Removes @mentions (e.g. @username)
    - Collapses multiple whitespace characters to a single space
    """
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove user mentions
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_preprocess_dataset(csv_path: str) -> pd.DataFrame:
    """
    Loads dataset, cleans text, normalizes labels, and drops exact duplicates
    prior to train-test splitting to ensure strict leak-free partitioning.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not locate training dataset: {csv_path}")

    print(f"[*] Loading training corpus from: {csv_path}")
    raw_df = pd.read_csv(csv_path)
    print(f"[*] Raw dataset row count: {len(raw_df):,}")

    df = raw_df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).apply(clean_corpus_text)
    df = df[df["text"].str.strip() != ""].copy()

    # Normalize labels strictly to 'Positive', 'Negative', 'Neutral'
    valid_labels = {"Positive", "Negative", "Neutral"}
    df["label"] = df["label"].astype(str).str.strip().str.capitalize()
    df = df[df["label"].isin(valid_labels)].copy()

    # Drop exact duplicate texts before splitting
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print(f"[+] Cleaned & Deduplicated dataset: {len(df):,} unique records")
    print(f"[+] Class distribution:\n{df['label'].value_counts().to_string()}\n")

    return df


def train_and_evaluate_pipelines():
    """
    Performs leak-free preprocessing, stratified 80-20 train-test splitting,
    trains baseline model, tunes Classical Upgrade pipeline with 5-fold Stratified GridSearchCV,
    and evaluates both on the untouched test partition.
    """
    global baseline_pipeline, upgraded_pipeline, MODEL_METRICS

    csv_file = DATA_PATH if os.path.exists(DATA_PATH) else "hinglish_data.csv"
    df = load_and_preprocess_dataset(csv_file)

    # 1. Stratified 80-20 train/test split (no data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )
    print(f"[*] Stratified split: {len(X_train):,} train samples | {len(X_test):,} hold-out test samples")
    print(f"[*] Training Class Distribution (Natural distribution, no synthetic oversampling):\n{y_train.value_counts().to_string()}\n")

    # 2. Train Baseline Pipeline (Word TF-IDF + Logistic Regression)
    print("[*] Training Baseline Pipeline (Word TF-IDF + Logistic Regression)...")
    base_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_features=10000,
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    base_pipe.fit(X_train, y_train)

    # 3. Setup Classical Upgrade Pipeline Template for GridSearchCV
    up_pipe_template = Pipeline([
        ('normalizer', HinglishNormalizer()),
        ('negation_tagger', HinglishNegationTagger()),
        ('features', FeatureUnion([
            ('word_tfidf', TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True
            )),
            ('char_tfidf', TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(3, 5),
                min_df=3,
                max_features=30000,
                sublinear_tf=True
            ))
        ])),
        ('clf', LinearSVC(class_weight='balanced', random_state=42, max_iter=2000))
    ])

    # 4. Hyperparameter Tuning via 3-Fold Stratified GridSearchCV
    param_grid = {
        'clf__C': [0.1, 0.5, 1.0]
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(
        up_pipe_template,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )
    print("[*] Tuning Classical Upgrade Pipeline with 3-Fold Stratified GridSearchCV (scoring='f1_macro', n_jobs=-1)...")
    grid.fit(X_train, y_train)

    print("\n" + "=" * 65)
    print("GRIDSEARCHCV HYPERPARAMETER OPTIMIZATION COMPLETE:")
    print("=" * 65)
    print(f"  Best 3-Fold CV Macro F1 : {grid.best_score_*100:.2f}%")
    print("  Best Hyperparameters    :")
    for param, val in grid.best_params_.items():
        print(f"    - {param}: {val}")
    print("=" * 65 + "\n")

    up_pipe = grid.best_estimator_

    # 5. Evaluate on untouched X_test and y_test
    base_preds = base_pipe.predict(X_test)
    base_acc = accuracy_score(y_test, base_preds)
    base_f1 = f1_score(y_test, base_preds, average='macro')

    up_preds = up_pipe.predict(X_test)
    up_acc = accuracy_score(y_test, up_preds)
    up_f1 = f1_score(y_test, up_preds, average='macro')

    # Store metrics in memory
    MODEL_METRICS = {
        "baseline": {
            "name": "Baseline (Word TF-IDF + LogReg)",
            "accuracy": round(base_acc * 100, 2),
            "macro_f1": round(base_f1 * 100, 2),
            "test_samples": len(X_test)
        },
        "upgraded": {
            "name": "Classical Upgrade (LinearSVC + Negation)",
            "accuracy": round(up_acc * 100, 2),
            "macro_f1": round(up_f1 * 100, 2),
            "test_samples": len(X_test)
        }
    }

    print("\n" + "=" * 65)
    print("FINAL HOLD-OUT TEST EVALUATION METRICS (Untouched 20% Partition):")
    print("=" * 65)
    print(f"  [1] Baseline Pipeline : Accuracy = {base_acc*100:.2f}% | Macro F1 = {base_f1*100:.2f}%")
    print(f"  [2] Tuned Classical Upgrade : Accuracy = {up_acc*100:.2f}% | Macro F1 = {up_f1*100:.2f}%")
    print(f"  Macro F1 Gain         : {(up_f1 - base_f1)*100:+.2f}%")
    print("=" * 65)

    print("\n" + "-" * 55)
    print("BASELINE MODEL CLASSIFICATION REPORT:")
    print("-" * 55)
    print(classification_report(y_test, base_preds))

    print("-" * 55)
    print("TUNED CLASSICAL UPGRADE CLASSIFICATION REPORT:")
    print("-" * 55)
    print(classification_report(y_test, up_preds))
    print("-" * 55 + "\n")

    # Persist pipelines to disk
    joblib.dump(base_pipe, BASELINE_PATH)
    joblib.dump(up_pipe, UPGRADED_PATH)
    print(f"[+] Persisted baseline pipeline to: {BASELINE_PATH}")
    print(f"[+] Persisted tuned upgraded pipeline to: {UPGRADED_PATH}")

    baseline_pipeline = base_pipe
    upgraded_pipeline = up_pipe


@app.on_event("startup")
def startup_event():
    global baseline_pipeline, upgraded_pipeline, MODEL_METRICS

    # Clean legacy single-model files if present
    for old_file in ["model.pkl", "vectorizer.pkl"]:
        p = os.path.join(BACKEND_DIR, old_file)
        if os.path.exists(p):
            try:
                os.remove(p)
                print(f"[-] Removed legacy artifact: {old_file}")
            except Exception as e:
                print(f"[!] Warning removing {old_file}: {e}")

    # Check if both pipelines are already persisted
    if os.path.exists(BASELINE_PATH) and os.path.exists(UPGRADED_PATH):
        try:
            print("[*] Found persisted pipelines on disk. Loading via Joblib...")
            baseline_pipeline = joblib.load(BASELINE_PATH)
            upgraded_pipeline = joblib.load(UPGRADED_PATH)
            print("[+] Successfully loaded both pipelines from disk.")

            # Load clean dataset to populate hold-out metrics
            df = load_and_preprocess_dataset(DATA_PATH if os.path.exists(DATA_PATH) else "hinglish_data.csv")
            _, X_test, _, y_test = train_test_split(
                df["text"], df["label"], test_size=0.20, random_state=42, stratify=df["label"]
            )
            base_p = baseline_pipeline.predict(X_test)
            up_p = upgraded_pipeline.predict(X_test)

            MODEL_METRICS = {
                "baseline": {
                    "name": "Baseline (Word TF-IDF + LogReg)",
                    "accuracy": round(accuracy_score(y_test, base_p) * 100, 2),
                    "macro_f1": round(f1_score(y_test, base_p, average='macro') * 100, 2),
                    "test_samples": len(X_test)
                },
                "upgraded": {
                    "name": "Classical Upgrade (LinearSVC + Negation)",
                    "accuracy": round(accuracy_score(y_test, up_p) * 100, 2),
                    "macro_f1": round(f1_score(y_test, up_p, average='macro') * 100, 2),
                    "test_samples": len(X_test)
                }
            }
            return
        except Exception as err:
            print(f"[!] Could not load persisted pipelines ({err}). Triggering fresh training...")

    # If pipelines are absent or failed to load, train them fresh
    train_and_evaluate_pipelines()


# The Enhanced Perturbation Engine (Lexical Shorthand + Phonetic Typo Noise)
def generate_variations(text: str, num_variants: int = 4):
    """
    Generates realistic Hinglish perturbations by combining:
    1. Lexical shorthand swaps based on Indian texting conventions
       (e.g., 'main' -> 'm'/'mein', 'bahut' -> 'bht'/'bohot', 'hai' -> 'h', 'achha' -> 'achi',
        'sab' -> 'sb', 'kab' -> 'kb', 'kaha' -> 'kha'/'kidhar', 'kaise' -> 'kse', 'kyun' -> 'kyu').
    2. Phonetic regular expression mutations (e.g., dropping soft 'h', vowel shortening, consonant swapping).
    """
    if not text or not text.strip():
        return []

    words = text.split()
    variants = []
    regex_rules = [
        (r'oo', 'u'), (r'ee', 'i'), (r'au', 'o'),
        (r'ph', 'f'), (r'bh', 'b'), (r'kh', 'k'),
        (r'h', ''), (r'([bcdfghjklmnpqrstvwxyz])a([bcdfghjklmnpqrstvwxyz])', r'\1\2')
    ]

    max_attempts = max(num_variants * 6, 25)
    attempts = 0

    while len(variants) < num_variants and attempts < max_attempts:
        attempts += 1
        new_sentence = []

        for word in words:
            # Separate surrounding punctuation from core word
            prefix_match = re.match(r'^\W+', word)
            suffix_match = re.search(r'\W+$', word)
            prefix = prefix_match.group(0) if prefix_match else ''
            suffix = suffix_match.group(0) if suffix_match else ''
            core = word[len(prefix):len(word) - len(suffix)] if suffix else word[len(prefix):]

            if not core:
                new_sentence.append(word)
                continue

            core_lower = core.lower()
            canonical_root = HINGLISH_CANONICAL_MAP.get(core_lower, core_lower)

            # Step 1: Lexical shorthand perturbation
            if canonical_root in LEXICAL_PERTURBATIONS and random.random() > 0.30:
                candidates = [v for v in LEXICAL_PERTURBATIONS[canonical_root] if v != core_lower]
                if candidates:
                    chosen = random.choice(candidates)
                    if core.isupper():
                        core = chosen.upper()
                    elif core.istitle():
                        core = chosen.capitalize()
                    else:
                        core = chosen

            # Step 2: Character-level phonetic perturbation
            elif len(core) > 3 and random.random() > 0.40:
                rule = random.choice(regex_rules)
                new_core = re.sub(rule[0], rule[1], core, count=1)
                if new_core != core:
                    core = new_core

            new_sentence.append(prefix + core + suffix)

        variant_candidate = " ".join(new_sentence)
        if variant_candidate != text and variant_candidate not in variants:
            variants.append(variant_candidate)

    # Fallback to force at least one variation if all probabilistic passes produced identical text
    if not variants:
        forced_sentence = []
        for word in words:
            core_lower = re.sub(r'^\W+|\W+$', '', word).lower()
            canonical_root = HINGLISH_CANONICAL_MAP.get(core_lower, core_lower)
            if canonical_root in LEXICAL_PERTURBATIONS:
                candidates = [v for v in LEXICAL_PERTURBATIONS[canonical_root] if v != core_lower]
                if candidates:
                    word = random.choice(candidates)
            forced_sentence.append(word)
        forced_candidate = " ".join(forced_sentence)
        if forced_candidate != text:
            variants.append(forced_candidate)

    return variants


# Analysis Request payload
class AnalysisRequest(BaseModel):
    sentence: str
    model_type: str = "upgraded"  # "upgraded" or "baseline"


@app.get("/metrics")
def get_metrics():
    """
    Returns hold-out test metrics (Accuracy, Macro F1) for both models.
    """
    return MODEL_METRICS


@app.post("/analyze")
def analyze_robustness(req: AnalysisRequest):
    original_text = req.sentence
    model_choice = req.model_type.lower().strip() if req.model_type else "upgraded"

    # Select requested pipeline
    if model_choice == "baseline":
        active_pipeline = baseline_pipeline
        model_label = "Baseline (Word TF-IDF + LogReg)"
    else:
        active_pipeline = upgraded_pipeline
        model_label = "Classical Upgrade (NegTag + Word/Char TF-IDF + LinearSVC)"

    if active_pipeline is None:
        return {"error": "Model pipelines are not initialized yet."}

    # 1. Predict baseline on original clean text
    baseline_pred = str(active_pipeline.predict([original_text])[0])

    # 2. Generate noisy variations
    noisy_texts = generate_variations(original_text, num_variants=4)

    # 3. Predict on variations & compute consistency score
    results = []
    consistent_count = 0

    if noisy_texts:
        variant_preds = active_pipeline.predict(noisy_texts)
        for text, pred in zip(noisy_texts, variant_preds):
            p_str = str(pred)
            if p_str == baseline_pred:
                consistent_count += 1
            results.append({
                "text": text,
                "pred": p_str
            })

    consistency_score = int((consistent_count / len(noisy_texts)) * 100) if noisy_texts else 100

    # 4. Generate negation-tagged version for live inspection
    negation_tagged_preview = tag_hinglish_negation(original_text)

    return {
        "original": original_text,
        "baseline_pred": baseline_pred,
        "consistency_score": consistency_score,
        "variations": results,
        "model_used": model_label,
        "negation_tagged": negation_tagged_preview
    }
