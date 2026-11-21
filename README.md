<<<<<<< HEAD
# 🛡️ Hinglish Robustness Evaluator: Assessing NLP Prediction Consistency Against Spelling Variations

**MCA 3rd-Semester NLP Research Project**  
*Evaluating and resolving the fragility of token-based text classification models on code-mixed, orthographically inconsistent Hindi-English (Hinglish) text.*

---

## 📌 Problem Statement

Code-mixed Hindi-English (popularly known as **Hinglish**) is one of the most widely used communication mediums across digital platforms in South Asia. However, unlike standard English or Devanagari Hindi, Hinglish **lacks standardized spelling and orthographic rules**. Users phonetically transcribe spoken words using the Latin alphabet according to their personal typing habits, dialectal background, or convenience:

- **Example**: The word *बहुत* ("very") is written arbitrarily as:
  - `bahut`, `bohot`, `bht`, `boht`, `bahot`, `bahuuut`, `bhut`
- **Example**: The negation word *नहीं* ("not") is written as:
  - `nahi`, `nhi`, `nahin`, `nai`, `nh`, `ni`, `nhn`
- **Example**: The interrogative *कहाँ* ("where") and *कैसे* ("how"):
  - `kaha`, `kha`, `kidhar`, `kdhr` | `kaise`, `kse`, `kese`

### Why Standard NLP Models Fail
Traditional Natural Language Processing (NLP) pipelines rely on exact lexical tokenization—such as word n-grams (e.g., standard TF-IDF) or rigid vocabularies. These tokenizers treat each variation as an **entirely separate token** or an **Out-Of-Vocabulary (OOV)** term. 

Furthermore, Hindi and Hinglish follow a **Subject-Object-Verb (SOV)** syntactic structure where negation particles frequently appear *after* the verb or adjective (e.g., `"movie achhi nahi thi"`), whereas English follows **SVO** (`"movie was not good"`). Standard unigrams and forward-only bigrams fail to associate negation with the sentiment-bearing adjective, causing severe misclassifications.

Because the model lacks phonetic awareness and subword resilience:
1. A slight character substitution or vowel omission drastically alters the extracted feature representation.
2. A sentiment classification model may classify `"yeh movie bahut achhi hai"` as **Positive**, yet flip to **Negative** or **Neutral** when evaluated on `"yeh movie bht achi h"`.
3. Inverting sentiment with informal negation (e.g., `"achha nhi tha"`) fails because `"achha"` and `"nhi"` are separated or treated as independent positive/negative unigrams.

---

## 🚀 The Classical Upgrade Architecture

To directly overcome these vulnerabilities without incurring the high computational and latency overhead of heavy deep neural networks, this project designs and benchmarks a **Classical NLP Upgrade Pipeline** implemented via custom `scikit-learn` transformers and dual-branch subword feature unions.

```
                                  ┌── Raw Hinglish Text ──┐
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │    HinglishNormalizer     │
                    │  (Elongation Reduction &  │
                    │   Canonical Lexicon Map)  │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │   HinglishNegationTagger  │
                    │ (Bidirectional SOV Window │
                    │   NOT_ Prefix Binding)    │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │       FeatureUnion        │
                    │ ├── Word TF-IDF (1, 2)    │
                    │ └── Char-wb TF-IDF (3, 5) │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    Linear Support Vector Classifier
                    LinearSVC(class_weight='balanced')
```

### 1. Hinglish Canonicalization Lexicon & Custom Transformers (`backend/nlp_utils.py`)
- **`HINGLISH_CANONICAL_MAP`**:
  A standardized lexicon of **117 canonical mappings** converting informal Hinglish spelling variants and ubiquitous Indian texting shorthand to canonical lemmas using strict whole-word regex matching (preventing substring corruption in words like *someone* or *more*):
  - **Pronouns**: `main` (`m`, `me`, `mein`, `mei`, `mai`), `mujhe` (`mjhe`, `muje`, `mujhko`, `mjko`), `tum` (`tm`, `tume`), `aap` (`ap`), `yeh` (`ye`, `yh`), `woh` (`wo`, `vo`, `voh`), `hum` (`hm`)
  - **Questions & Interrogatives**: `kaha` (`kha`, `kahan`, `kahaan`, `kidhar`, `kidhr`, `kdhr`), `kaise` (`kse`, `kese`, `kaisse`), `kyun` (`kyu`, `kyon`, `kyn`, `q`), `kaun` (`kon`, `kn`), `kab` (`kb`, `kbb`)
  - **Quantifiers & Adverbs**: `sab` (`sb`, `sabb`), `abhi` (`abi`, `abhii`), `kal` (`kl`), `aaj` (`aj`)
  - **Verbs & Auxiliaries**: `kar` (`kr`), `karo` (`kro`), `karna` (`krna`), `raha` (`rha`), `rahi` (`rhi`), `rahe` (`rhe`), `hai` (`h`, `he`, `hn`, `hain`, `hyn`), `kya` (`ky`, `kyaa`, `kia`), `kyunki` (`kyuki`, `kyoki`, `qki`), `bhi` (`bi`, `bhii`), `aur` (`or`), `kuch` (`kch`, `kuchh`)
  - **Intensifiers**: `bahut` (`bht`, `bohot`, `boht`, `bahoot`, `bahot`, `bhut`), `bilkul` (`blkl`, `bilkool`, `bilkull`), `ekdum` (`ekdam`, `ekdm`, `ikdam`), `zyada` (`jyada`, `jada`, `zyaada`, `zyda`), `thoda` (`thora`, `thda`)
  - **Sentiment Words**: `achha` (`acha`, `accha`, `achi`, `acchi`, `achhi`, `achchi`), `thik` (`theek`, `thk`, `tk`), `bakwas` (`bkwas`, `bakwaas`, `bkwass`, `bakvas`), `bekar` (`bekaar`, `bkr`, `bkaar`), `ghatiya` (`gatiya`, `ghatia`, `ghatiyya`), `mast` (`mst`, `masst`), `zabardast` (`zbrdst`, `jabardast`, `jbrdst`)
  - **Fillers & Slang**: `yaar` (`yr`, `yar`, `yaara`), `bhai` (`bhaai`, `bhay`)
  - **Negations**: `nahi` (`nhi`, `nai`, `nahin`, `nh`, `nhe`, `nhn`), `na` (`naa`), `mat` (`mt`)

- **`HinglishNormalizer`**:
  - Collapses excessive character elongations (e.g., `bahuuuut` $\rightarrow$ `bahut`, `sooo` $\rightarrow$ `soo`) to prevent vocabulary explosion while retaining phonetic semantics.
  - Normalizes text against `HINGLISH_CANONICAL_MAP` to ensure all informal SMS shorthand variants collapse to standardized roots.

- **`HinglishNegationTagger`**:
  - Implements a bidirectional negation window ($W=3$) tailored specifically for Indic SOV word order.
  - Standardizes tokens and prepends a `NOT_` prefix to neighboring words within the window (e.g., `"movie achhi nahi thi"` $\rightarrow$ `"movie NOT_achha nahi NOT_thi"`).
  - Stops negation propagation across clause boundaries (periods, commas, exclamation marks, question marks, semicolons).
  - Allows simple linear classifiers to separate negated sentiment words from affirmative ones cleanly.

### 2. Dual-Branch Feature Union
- **Branch A (Word N-Grams + Negation)**: Word-level TF-IDF (`ngram_range=(1, 2)`, `sublinear_tf=True`, `min_df=2`) captures syntactic phrases and explicit negation tokens.
- **Branch B (Subword Character N-Grams)**: Character word-boundary n-grams (`analyzer='char_wb'`, `ngram_range=(3, 5)`, `sublinear_tf=True`, `min_df=3`, `max_features=30000`) captures internal root morphemes, allowing the model to effortlessly recognize words with omitted vowels or typos (e.g., `bkwass` $\leftrightarrow$ `bakwas`, `brbad` $\leftrightarrow$ `barbad`).

### 3. Classification Engine
- **`LinearSVC`**: Utilizes a Linear Support Vector Classifier with balanced class weights (`class_weight='balanced'`) and regularized penalty ($C=0.1$) to maximize margin separation across the sparse, high-dimensional joint subword feature space and prevent majority-class bias.

---

## 📈 Quantitative Benchmark & Evaluation Results

Both models were evaluated on the leak-free, untouched stratified 20% hold-out test set ($N_{\text{total}} = 17,148$, $N_{\text{train}} = 13,718$, $N_{\text{test}} = 3,430$) from the expanded multi-source corpus `backend/hinglish_data.csv`:

### Hold-out Test Performance ($N_{\text{test}} = 3,430$)

| Evaluation Metric | Baseline (Word TF-IDF + LogReg) | Tuned Classical Upgrade (Union + LinearSVC) | Improvement ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Hold-out Test Accuracy** | 61.17% | **62.86%** | **+1.69%** |
| **Macro Precision** | 61.08% | **62.75%** | **+1.67%** |
| **Macro Recall** | 61.51% | **63.22%** | **+1.71%** |
| **Macro F1-Score** | 61.21% | **62.85%** | **+1.65%** |
| **Weighted F1-Score** | 60.94% | **62.59%** | **+1.65%** |
| **Negative Class F1 / R / P** | 0.6615 / 0.6966 / 0.6297 | **0.6744 / 0.7157 / 0.6375** | **+1.29% F1 (+1.91% Recall)** |
| **Neutral Class F1 / R / P** | 0.5176 / 0.4946 / 0.5427 | **0.5350 / 0.5062 / 0.5673** | **+1.75% F1 (+1.16% Recall)** |
| **Positive Class F1 / R / P** | 0.6571 / 0.6542 / 0.6601 | **0.6762 / 0.6747 / 0.6777** | **+1.91% F1 (+2.05% Recall)** |

### Adversarial Perturbation Benchmark ($N_{\text{samples}} = 500$, $N_{\text{variants}} = 2,000$)

| Robustness Metric | Baseline (Word TF-IDF + LogReg) | Tuned Classical Upgrade (Union + LinearSVC) | Improvement ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Clean Accuracy** | 62.00% | **62.80%** | **+0.80%** |
| **Prediction Consistency Score (%)** | 91.40% | **93.60%** | **+2.20%** |
| **Perturbed Accuracy** | 61.30% | **63.05%** | **+1.75%** |

*All comparative metrics, confusion matrix heatmaps, cross-validation tables, and robustness visualizations are computed and persisted in `experiments/classical_upgrade_benchmark.ipynb` and saved in `figures/`.*

---

## 🛡️ Preprocessing, Label De-Noising & Hyperparameter Tuning Pipeline

To maximize academic rigor, prevent data leakage, and ensure clean labels:
1. **Multi-Source Aggressive Noise & Label Cleaning (`backend/get_real_data.py`)**:
   - Strips URLs (`https?://...`, `www...`), `@mentions`, `RT` prefixes, and extra whitespace; strips `#` symbols while preserving the hashtag word.
   - Drops rows with fewer than 3 alphabetic words or containing only numbers, emojis, and symbols.
   - Normalizes sentiment labels strictly to `Positive`, `Negative`, and `Neutral`.
   - **Conflicting Label Removal**: Groups by lowercased, stripped text. If the exact same sentence appears with differing sentiment labels (annotator conflict/disagreement across datasets), **all conflicting rows are discarded**.
   - Drops exact duplicates, resulting in **17,148 high-quality unique samples**.
2. **Elimination of Double-Balancing**:
   - Replaced artificial training upsampling (`resample`) with natural class distribution training, relying solely on `class_weight='balanced'` inside `LinearSVC`. This avoids overfitting to duplicated sentences while maintaining equal penalization for minority misclassifications.
3. **3-Fold Stratified Hyperparameter Optimization (`GridSearchCV`)**:
   - Evaluated regularization parameter using `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` on $X_{\text{train}}$ (13,718 samples) scored on `f1_macro` (`n_jobs=-1`):
     - `clf__C`: `[0.1, 0.5, 1.0]`
   - **Optimal Parameter Discovered**:
     - `clf__C = 0.1` (regularization preventing overfit on sparse Hinglish text features)
   - **Best CV Macro F1**: **61.38%** (0.6138) across 3-fold cross-validation.

---

## 📊 Dataset: Multi-Source Real-World Hinglish Sentiment Corpus

The project merges and cleans multiple prominent public Hinglish datasets into a unified benchmark corpus:

- **Primary Data Sources (Hugging Face Hub)**:
  1. **`shae2977/hinglish-youtube-sentiments-dataset`**: 3,129 clean user comments across YouTube videos in Romanized, code-mixed Hindi-English.
  2. **`Abhishek4896/hindi-english-code-mixed-tweets-sentiment`**: 498 conversational tweets reflecting microblogging slang and abbreviations.
  3. **`RTT1/SentiMix`**: 13,999 sentences parsed from the official SemEval-2020 Task 9 (SentiMix Hinglish) shared task corpus.
- **Combined Corpus Statistics (`backend/hinglish_data.csv`)**:
  - **Total Merged Raw Rows**: 17,626
  - **Noise & Conflict Rows Dropped**: 478
  - **Final Cleaned Rows Saved**: **17,148 unique records**
  - **Class Distribution (Naturally Balanced)**:
    - **Neutral**: 6,032 records (35.18%)
    - **Positive**: 5,609 records (32.71%)
    - **Negative**: 5,507 records (32.11%)

---

## 🏛️ System Architecture

The application is structured into a decoupled client-server architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                 React + Tailwind CSS Frontend               │
│  • Model Toggle: [Classical Upgrade ★] vs [Baseline]        │
│  • Hold-out Benchmark Metric Banner (Acc & Macro F1)        │
│  • Negation-Tagged Scope Preview Box (NOT_ Token Chips)     │
│  • Consistency Score Gauge & Perturbation Data Table        │
└──────────────────────────────┬──────────────────────────────┘
                               │  POST /analyze { sentence, model_type }
                               │  GET  /metrics
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ 1. Lifespan Dual Pipeline Loader / Retrainer          │  │
│  │    • baseline_pipeline.pkl (TF-IDF + LogReg)          │  │
│  │    • upgraded_pipeline.pkl (FeatureUnion + LinearSVC) │  │
│  ├───────────────────────────────────────────────────────┤  │
│  │ 2. Custom NLP Transformers (backend/nlp_utils.py)     │  │
│  │    • HinglishNormalizer & HinglishNegationTagger      │  │
│  ├───────────────────────────────────────────────────────┤  │
│  │ 3. Perturbation Engine (Synthetic Noise Injection)    │  │
│  ├───────────────────────────────────────────────────────┤  │
│  │ 4. Evaluation Engine (Holdout Accuracy & Macro F1)    │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Backend Endpoints
- **`POST /analyze`**:
  - **Request Body**: `{ "sentence": string, "model_type": "upgraded" | "baseline" }`
  - **Response Payload**:
    ```json
    {
      "original": "bhai ekdum bkwass movie thi,paisa brbad",
      "model_used": "Classical Upgrade (NegTag + Word/Char TF-IDF + LinearSVC)",
      "negation_tagged": "bhai ekdum bkwass movie thi,paisa brbad",
      "baseline_pred": "Negative",
      "consistency_score": 100.0,
      "variations": [
        { "text": "bai ekdum bkwass movie thi,paisa brbad", "pred": "Negative" },
        { "text": "bhai ekdum bkwass movie ti,paisa brbad", "pred": "Negative" }
      ]
    }
    ```
- **`GET /metrics`**:
  - Returns hold-out test accuracy, macro F1, and test partition counts for both pipelines.

---

## ⚙️ How it Works: The Enhanced Dual-Mode Perturbation Engine

The **Perturbation Engine** (`generate_variations` in `backend/main.py`) acts as an adversarial synthetic noise injection framework that simultaneously simulates **cognitive Indian texting shorthand** and **phonological orthographic drift**:

### 1. Mode A: Lexical Shorthand Swapping
Leveraging `LEXICAL_PERTURBATIONS`, the engine identifies canonical Hinglish terms and probabilistically swaps them with authentic conversational chat shorthand:
- **Pronouns**: `main` $\rightarrow$ `m`, `me`, `mein`, `mai` | `yeh` $\rightarrow$ `ye`, `yh` | `mujhe` $\rightarrow$ `mjhe`, `mjko`
- **Questions & Interrogatives**: `kaha` $\rightarrow$ `kha`, `kidhar` | `kaise` $\rightarrow$ `kse`, `kese` | `kyun` $\rightarrow$ `kyu`, `q` | `kab` $\rightarrow$ `kb` | `kaun` $\rightarrow$ `kon`
- **Quantifiers & Auxiliaries**: `sab` $\rightarrow$ `sb` | `abhi` $\rightarrow$ `abi` | `kal` $\rightarrow$ `kl` | `aaj` $\rightarrow$ `aj` | `kar`/`karna` $\rightarrow$ `kr`/`krna` | `raha`/`rahe` $\rightarrow$ `rha`/`rhe` | `mat` $\rightarrow$ `mt`
- **Verbs & Connectors**: `hai` $\rightarrow$ `h`, `he`, `hain` | `aur` $\rightarrow$ `or` | `bhi` $\rightarrow$ `bi`
- **Intensifiers & Modifiers**: `bahut` $\rightarrow$ `bht`, `bohot`, `bhut` | `bilkul` $\rightarrow$ `blkl` | `ekdum` $\rightarrow$ `ekdm`
- **Sentiment & Negations**: `achha` $\rightarrow$ `achi`, `accha` | `bakwas` $\rightarrow$ `bkwas`, `bkwass` | `nahi` $\rightarrow$ `nhi`, `nai`, `nh`
- **Fillers**: `yaar` $\rightarrow$ `yr`, `yar` | `bhai` $\rightarrow$ `bhay`, `bhaai`

### 2. Mode B: Phonetic Regex Typo Mutations
For non-lexicon terms or longer tokens ($>3$ characters), the engine applies phonological regular expression transformations:

| Pattern Type | Rule / Regex | Example Clean Word | Perturbed Variation |
| :--- | :--- | :--- | :--- |
| **Vowel Flattening** | `oo` $\rightarrow$ `u` | `khoobsurat` | `khubsurat` |
| **Long Vowel Shortening** | `ee` $\rightarrow$ `i` | `theek` | `thik` |
| **Diphthong Reduction** | `au` $\rightarrow$ `o` | `aur` | `or` |
| **Consonant De-aspiration** | `ph` $\rightarrow$ `f` | `phir` | `fir` |
| **Aspiration Dropping** | `bh` $\rightarrow$ `b`, `kh` $\rightarrow$ `k` | `bhai`, `khana` | `bai`, `kana` |
| **Silent / Soft 'h' Omission** | `h` $\rightarrow$ `""` | `bohot` | `boot` |
| **Medial Vowel Omission** | `[consonant]a[consonant]` $\rightarrow$ `[c][c]` | `khatam` | `khtm` |

By combining both modes, the engine tests whether an NLP model evaluates meaning robustly when exposed to modern conversational typing habits.

### Quantitative Consistency Metric
Given an original sentence $S$ with baseline prediction $\hat{y} = M(S)$ and a set of $N$ synthetic variations $\{v_1, v_2, \dots, v_N\}$ generated by the perturbation engine:

$$\text{Consistency Score} = \left( \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left(M(v_i) == \hat{y}\right) \right) \times 100\%$$

---

## 📁 Project Directory Structure

```text
Hinglish_Robustness_Project/
├── README.md                              # Complete research documentation & benchmark report
├── backend/
│   ├── get_real_data.py                   # Data ingestion: Hugging Face multi-source downloader
│   ├── nlp_utils.py                       # Custom Sklearn transformers (Normalizer, NegationTagger)
│   ├── main.py                            # FastAPI server, leak-free pipeline, & endpoints
│   ├── test_api.py                        # Automated backend test suite (FastAPI TestClient)
│   ├── hinglish_data.csv                  # Cleaned & deduplicated corpus (17,148 unique rows)
│   ├── baseline_pipeline.pkl              # Persisted Baseline Pipeline (TF-IDF + LogReg)
│   └── upgraded_pipeline.pkl              # Persisted Classical Upgrade (FeatureUnion + LinearSVC)
├── experiments/
│   └── classical_upgrade_benchmark.ipynb  # Comparative benchmark notebook (metrics & plots)
└── frontend/
    ├── package.json                       # Dependencies & build scripts (Vite, Tailwind v4, React 19)
    ├── vite.config.js                     # Vite build configuration
    └── src/
        ├── index.css                      # Tailwind v4 import entry
        └── App.jsx                        # Upgraded Research Dashboard (Toggle, Banner, Negation Chips)
```

---

## 🧪 Automated API Testing Suite

The backend includes an automated test harness in `backend/test_api.py` powered by FastAPI's `TestClient` to validate endpoints, schemas, and prediction consistency without running a live HTTP server manually:

```bash
python backend/test_api.py
```

### What the Test Suite Verifies:
1. **Startup Lifecycle**: Confirms both pipelines load from disk and recalculate hold-out test metrics on the untouched test partition.
2. **`GET /metrics` Verification**: Asserts HTTP 200 and checks that hold-out Test Accuracy and Macro F1 are accurately served for both models.
3. **`POST /analyze` Regression Checks**: Evaluates both the `upgraded` and `baseline` models across critical Hinglish edge cases:
   - Positive slang & intensifiers (`"m bahut khush hu, movie ekdum mast thi"`)
   - Negation scoping (`"Khana bilkul thik nhi laga mujhe"`)
   - Negative colloquial slang (`"bhai wo banda ekdum badtamiz hai, bilkul bakwas behaviour"`)
   - Neutral code-mixed plans (`"sb log kb aur kha milenge, average plan h"`)
4. **Consistency Comparison**: Asserts HTTP 200 responses and prints the prediction status for every generated perturbation (demonstrating how the baseline flips predictions on noisy shorthand while the Classical Upgrade remains stable).

---

## 🚀 Getting Started

### Prerequisites
- **Python**: Version 3.10 or higher
- **Node.js**: Version 18.0.0 or higher with `npm`

---

### Step 1: Backend Setup & Server Launch

1. Open a terminal and navigate to the project root:
   ```bash
   cd Hinglish_Robustness_Project
   ```

2. Install backend dependencies:
   ```bash
   pip install fastapi uvicorn scikit-learn pandas pydantic datasets joblib
   ```

3. (Optional) Re-fetch or verify the combined Hugging Face dataset:
   ```bash
   python backend/get_real_data.py
   ```

4. Start the FastAPI development server:
   ```bash
   python -m uvicorn backend.main:app --reload --port 8000
   ```
   *The server loads both persisted pipelines (or retrains them automatically if missing), displays holdout test metrics in the console, and listens at `http://127.0.0.1:8000`.*

---

### Step 2: Frontend Setup & Dashboard Launch

1. Open a second terminal window and navigate to `frontend`:
   ```bash
   cd frontend
   ```

2. Install packages:
   ```bash
   npm install
   ```

3. Launch Vite development server:
   ```bash
   npm run dev
   ```

4. Open `http://localhost:5173` in your browser.

---

## 🧪 Live Comparison & Qualitative Edge-Case Evaluation

To observe the real-world impact of the Classical Upgrade:

1. **Test Subword Robustness**:
   - Input: `"bhai ekdum bkwass movie thi,paisa brbad"`
   - Select **Baseline (Word TF-IDF + LogReg)** $\rightarrow$ The baseline misclassifies it as `Positive` because the informal spellings `bkwass` and `brbad` are OOV words.
   - Switch to **Classical Upgrade (LinearSVC + Negation)** $\rightarrow$ Correctly predicts `Negative` with high stability because character subword n-grams match the subword morphemes!

2. **Test SOV Negation Scope**:
   - Input: `"Khana bilkul thik nhi laga mujhe"`
   - Select **Baseline (Word TF-IDF + LogReg)** $\rightarrow$ Misclassifies as `Neutral` (and flips on spelling variants).
   - Switch to **Classical Upgrade (LinearSVC + Negation)** $\rightarrow$ Correctly predicts `Negative` because `NOT_thik` captures the inverted sentiment!
   - Observe the **Negation-Tagged Preview Box**: Notice how tokens adjacent to `nahi` are transformed into `NOT_achha` / `NOT_tha`, preventing the model from confusing the sentence with a positive review.

### Edge-Case Performance Breakdown
To evaluate real-world robustness beyond aggregate corpus metrics, both models were tested against 6 representative linguistic edge cases:

| Category | Input Sentence | Expected | Baseline (LogReg) | Tuned Upgrade (LinearSVC) | Empirical Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Information Query** | `"bhai review kab aayega movie ka? kidhar hai?"` | Neutral | Negative | **Neutral** | **✓ Upgrade Wins** (normalizer resolves `kab`/`kidhar` interrogatives) |
| **Positive Shorthand** | `"kya mst performance di h hero ne"` | Positive | Neutral | **Positive** | **✓ Upgrade Wins** (resolves `mst` $\to$ `mast` and `h` $\to$ `hai`) |
| **Double Negation** | `"aisa nahi hai ki movie bekar nahi thi"` | Negative | Negative | **Negative** | **✓ Both Pass** (both identify dominant `bekar` cue) |
| **Texting Shorthand** | `"m bht sad hu aj"` | Negative | Neutral | **Neutral** | **✗ Both Fail** (sparse context; weak weight on `sad` vs neutral prior) |
| **SOV Negation** | `"khana bilkul bhi swadist nahi tha"` | Negative | Neutral | **Neutral** | **✗ Both Fail** (`swadist` is low-frequency; `NOT_swadist` lacks negative prior) |
| **Slang & Phonetic** | `"yrr ye movie ekdm raddi thi"` | Negative | Negative | **Positive** | **✗ Baseline Wins (Upgrade Fails)** (`raddi` is OOV; char n-grams matched positive affixes) |

### ⚠️ Limitations & Error Analysis
While the domain-adapted classical pipeline significantly outperforms the baseline across hold-out generalization (+1.69% accuracy, +1.65% macro F1) and perturbation consistency (93.60% vs 91.40%), systematic error analysis reveals specific architectural boundaries:
1. **Low-Frequency Vocabulary & Sanskritized Hindi**: Words such as *swadist* (tasty) or *raddi* (trash/scrap) have very low frequency in social media comments compared to common tokens (*achha*, *bakwas*). When negation binds to an infrequent word (`NOT_swadist`), the model defaults to Neutral because the training set contains insufficient support for that negated feature.
2. **Subword Spurious Overlap**: For out-of-vocabulary slang terms (e.g., *raddi*), subword character n-grams can occasionally match subword character sequences in unrelated high-frequency positive words, leading to false-positive classification flips.
3. **Very Short Sentences & Context Starvation**: In sparse 4-to-5 word queries (e.g., *"m bht sad hu aj"*), the lack of strong contextual cues causes both models to be biased toward the majority prior (Neutral).

---

## 🎓 Academic Attribution
- **Degree**: Master of Computer Applications (MCA)
- **Course**: NLP / Machine Learning Laboratory & Research Seminar (Semester 3)
=======

