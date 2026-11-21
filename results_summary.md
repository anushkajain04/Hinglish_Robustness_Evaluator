# 📊 Hinglish Robustness Evaluator: Final Verified Benchmark & Consistency Report

**Project**: MCA 3rd-Semester NLP Research Project  
**Author**: Anushka Jain  
**Date**: November 2026  
**Status**: 100% Synchronized & Verified via Executed Code

---

## 1. Final Verified Benchmark Numbers (Hold-out Test Set, $N_{\text{test}} = 3,430$)

All numbers below are extracted directly from executed test outputs on the untouched 20% hold-out partition ($N_{\text{total}} = 17,148$, $N_{\text{train}} = 13,718$, $N_{\text{test}} = 3,430$, random seed = 42).

### A. Aggregate Hold-out Evaluation Metrics

| Metric | Baseline Pipeline (Word TF-IDF + LogReg) | Tuned Classical Upgrade (Union + LinearSVC) | Absolute Gain ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **61.17%** | **62.86%** | **+1.69%** |
| **Macro Precision** | **61.08%** | **62.75%** | **+1.67%** |
| **Macro Recall** | **61.51%** | **63.22%** | **+1.71%** |
| **Macro F1-Score** | **61.21%** | **62.85%** | **+1.65%** |
| **Weighted F1-Score** | **60.94%** | **62.59%** | **+1.65%** |

### B. Per-Class Precision, Recall & F1-Score

| Class | Support | Baseline Precision | Baseline Recall | Baseline F1 | Upgrade Precision | Upgrade Recall | Upgrade F1 | F1 Gain ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Negative** | 1,101 | 0.6297 (62.97%) | 0.6966 (69.66%) | 0.6615 (66.15%) | **0.6375 (63.75%)** | **0.7157 (71.57%)** | **0.6744 (67.44%)** | **+1.29%** |
| **Neutral** | 1,207 | 0.5427 (54.27%) | 0.4946 (49.46%) | 0.5176 (51.76%) | **0.5673 (56.73%)** | **0.5062 (50.62%)** | **0.5350 (53.50%)** | **+1.75%** |
| **Positive** | 1,122 | 0.6601 (66.01%) | 0.6542 (65.42%) | 0.6571 (65.71%) | **0.6777 (67.77%)** | **0.6747 (67.47%)** | **0.6762 (67.62%)** | **+1.91%** |

---

## 2. Hyperparameter Optimization & Grid Details

- **Dataset**: `backend/hinglish_data.csv` (17,148 unique clean records)
- **Validation Strategy**: 3-Fold Stratified K-Fold Cross-Validation on $X_{\text{train}}$ ($N=13,718$)
- **Scoring Function**: `f1_macro`
- **Parameter Grid**:
  - `clf__C`: `[0.1, 0.5, 1.0]`
- **Cross-Validation Results**:

| Candidate ($C$) | Mean CV Macro F1 | Std CV Macro F1 | Rank |
| :---: | :---: | :---: | :---: |
| **$C = 0.1$** | **0.6138 (61.38%)** | **0.0052** | **1 (Optimal)** |
| $C = 0.5$ | 0.5969 (59.69%) | 0.0052 | 2 |
| $C = 1.0$ | 0.5884 (58.84%) | 0.0073 | 3 |

- **Best Estimator Selection**: `LinearSVC(C=0.1, class_weight='balanced', random_state=42, max_iter=2000)`

---

## 3. Adversarial Perturbation & Robustness Analysis

- **Sample Size**: 500 randomly sampled hold-out test sentences ($N_{\text{variants}} = 2,000$ perturbed texts)
- **Perturbation Modes**: Vowel elongation, texting shorthand substitution, negation spacing distortion, and keyboard typo character swapping.

| Robustness Metric | Baseline Pipeline | Tuned Classical Upgrade | Absolute Difference ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Clean Test Accuracy** | 62.00% | **62.80%** | **+0.80%** |
| **Prediction Consistency Score (%)** | 91.40% | **93.60%** | **+2.20%** |
| **Perturbed Test Accuracy** | 61.30% | **63.05%** | **+1.75%** |

---

## 4. Qualitative Edge-Case Performance Breakdown

| Category | Input Sentence | Expected Label | Baseline Prediction | Tuned Upgrade Prediction | Empirical Evaluation Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Information Query** | `"bhai review kab aayega movie ka? kidhar hai?"` | Neutral | Negative | **Neutral** | **✓ Upgrade Wins** (Normalizer resolves interrogative `kab`/`kidhar`) |
| **Positive Shorthand** | `"kya mst performance di h hero ne"` | Positive | Neutral | **Positive** | **✓ Upgrade Wins** (Resolves `mst` $\to$ `mast` and `h` $\to$ `hai`) |
| **Double Negation** | `"aisa nahi hai ki movie bekar nahi thi"` | Negative | Negative | **Negative** | **✓ Both Pass** (Both capture the dominant `bekar` cue) |
| **Texting Shorthand** | `"m bht sad hu aj"` | Negative | Neutral | **Neutral** | **✗ Both Fail** (Sparse 5-word sentence; weak lexical weight on `sad`) |
| **SOV Negation** | `"khana bilkul bhi swadist nahi tha"` | Negative | Neutral | **Neutral** | **✗ Both Fail** (`swadist` is low-frequency; `NOT_swadist` lacks negative prior) |
| **Slang & Phonetic** | `"yrr ye movie ekdm raddi thi"` | Negative | **Negative** | Positive | **✗ Baseline Wins (Upgrade Fails)** (`raddi` is OOV; char n-grams matched positive affixes) |

---

## 5. Before vs After Comparison (Old Discrepant vs New Verified Numbers)

| Dimension | Previous / Discrepant Value in README | Verified Ground Truth (Notebook & Backend) | Root Cause of Previous Difference |
| :--- | :---: | :---: | :--- |
| **Baseline Test Accuracy** | 60.73% | **61.17%** | Old backend used raw `min_df=1, max_features=None, sublinear_tf=False`; notebook uses tuned `min_df=3, max_features=10000, sublinear_tf=True`. |
| **Upgraded Test Accuracy** | 63.35% | **62.86%** | Old backend used nested parallel branches inside `FeatureUnion`; notebook uses clean sequential pipeline (`Normalizer` $\to$ `Tagger` $\to$ `FeatureUnion`). |
| **Baseline Macro F1** | 59.97% | **61.21%** | Vectorizer parameter differences. |
| **Upgraded Macro F1** | 63.32% | **62.85%** | Sequential pipeline architecture difference. |
| **Best CV Macro F1** | 61.53% | **61.38%** | Old README reported 5-fold CV with 24 grid combinations; actual verified model runs 3-fold CV with $C \in [0.1, 0.5, 1.0]$. |
| **Perturbation Consistency** | 91.02% vs 96.64% | **91.40% vs 93.60%** | Old README numbers were from an unseeded synthetic run; new numbers are strictly measured on 500 test samples (2,000 variants) with seed 42. |
| **Perturbed Accuracy** | 61.93% vs 63.71% | **61.30% vs 63.05%** | Verified evaluation run. |
| **Negation Tagging Output** | `NOT_achhi` | `NOT_achha` | `tag_hinglish_negation` normalizes words first (`achhi` $\to$ `achha`) before prepending `NOT_`. |
| **Edge-Case Performance** | Stated as winning all cases | **2 Upgrade Wins, 1 Baseline Win, 1 Both Pass, 2 Both Fail** | Honest reporting of model limitations on rare slang and short context sentences. |

---

## 6. Detailed List of Changes

### A. `backend/main.py`
1. **Pipeline Configuration Synchronization**:
   - Updated `base_pipe` to use `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=10000, sublinear_tf=True)` matching the notebook.
   - Updated `up_pipe_template` to use the sequential architecture:
     `HinglishNormalizer()` $\to$ `HinglishNegationTagger()` $\to$ `FeatureUnion([word_tfidf(1,2, min_df=2, sublinear_tf=True), char_tfidf(3,5, min_df=3, max_features=30000, sublinear_tf=True)])` $\to$ `LinearSVC(class_weight='balanced', random_state=42, max_iter=2000)`.
2. **GridSearchCV Synchronization**:
   - Updated parameter grid to `clf__C: [0.1, 0.5, 1.0]` with 3-fold Stratified CV.
3. **Artifact Persistence**:
   - Re-executed `train_and_evaluate_pipelines()` and re-persisted `backend/baseline_pipeline.pkl` and `backend/upgraded_pipeline.pkl` with the synchronized models.
   - Verified that `GET /metrics` returns Accuracy = 61.17% (Baseline) and 62.86% (Upgrade).

### B. `experiments/classical_upgrade_benchmark.ipynb`
1. **Full Top-to-Bottom Execution**:
   - Re-executed all 23 cells top to bottom with seed 42.
   - Embedded real outputs, confusion matrix plots, and robustness bar charts into the notebook cells.
2. **Markdown Conclusion (Cell 22)**:
   - Synchronized all metrics in "8. Empirical Findings & Conclusions" to match the actual executed outputs (61.17% vs 62.86%, Macro F1 61.21% vs 62.85%, Consistency 91.40% vs 93.60%).
3. **Edge-Case Reporting (Cell 19)**:
   - Updated evaluation logic to explicitly distinguish when Baseline wins and Upgrade fails (`✗ Baseline Wins (Upgrade Fails)`).

### C. `README.md`
1. **Architecture Diagram**:
   - Updated ASCII flowchart to reflect the sequential pipeline (`Raw Text` $\to$ `HinglishNormalizer` $\to$ `HinglishNegationTagger` $\to$ `FeatureUnion` $\to$ `LinearSVC`).
2. **Tagging Representation**:
   - Corrected `NOT_achhi` to `NOT_achha`.
3. **Quantitative Metrics Tables**:
   - Updated the hold-out test performance table (Accuracy 61.17% vs 62.86%, Macro F1 61.21% vs 62.85%).
   - Updated the per-class classification metrics table.
   - Updated the adversarial perturbation benchmark table (Consistency 91.40% vs 93.60%, Perturbed Acc 61.30% vs 63.05%).
   - Updated the hyperparameter tuning section to reflect 3-fold CV with $C \in [0.1, 0.5, 1.0]$ and best CV Macro F1 = 61.38%.
4. **Qualitative Edge Cases & Limitations**:
   - Added the honest 6-case edge-case table.
   - Added a dedicated "Limitations & Error Analysis" section explaining failure modes on low-frequency vocabulary (*swadist*, *raddi*) and short, context-starved queries.

### D. `figures/`
- Generated and saved high-resolution PNG figures:
  - `figures/confusion_matrices.png`: Side-by-side hold-out test confusion matrices.
  - `figures/robustness_benchmark.png`: Grouped bar chart comparing Clean Accuracy, Consistency Score, and Perturbed Accuracy.
  - `figures/class_distribution.png`: Corpus class balance distribution.
