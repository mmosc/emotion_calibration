# Emotion-Aware Evaluation of Music Recommender Systems

This repository contains the code and experiments for an emotion-aware evaluation of music recommender systems. The project investigates the trade-off between ranking accuracy and emotional calibration, and adapts the CaliTune post-hoc calibration framework to a music recommendation setting.

## 1. Project Overview

Traditional recommender systems are typically optimized for ranking accuracy (e.g., nDCG), but ignore whether recommendations match a user’s emotional preferences.
This project extends standard evaluation by incorporating emotion-aware calibration metrics, and applies post-hoc greedy re-ranking inspired by the CaliTune paper.

**Main goals:**
- Evaluate standard recommender models using ranking metrics.
- Measure emotional calibration of recommendations.
- Apply CaliTune-style greedy post-processing to improve calibration.
- Analyze the trade-off between accuracy and calibration.

## 2. Data Preparation

### 2.1 Raw Listening History
- **Input file**: `data/listening_history.csv`
- Contains user–song interaction logs (play counts).
- **Data cleaning steps**:
    - Removed invalid or empty user/song entries.
    - Aggregated play counts per (user, song) pair.

### 2.2 Interaction Binarization
- Converted play counts into implicit feedback.
- **Workflow**:
    1. Kept only (user, song) pairs with **≥ 2 listens** to remove initial noise.
    2. Applied **5-core filtering** (iteratively removed users and songs with < 5 interactions).
- Assigned a fixed implicit feedback label (`label = 5`, compatible with RecBole).
- **Output file**: `outputs/01_preprocessing/interactions_binarized.csv`

### 2.3 Conversion to RecBole Format
- Converted interactions into RecBole-compatible `.inter` schema:
  ```
  user:token    item:token    label:float
  ```
- Enabled training and inference with RecBole models.

## 3. Emotion Metadata Processing

### 3.1 GEM Emotion Labels
- **Input file**: `data/id_gems.tsv`
- Contains multi-dimensional GEM emotion scores per song.

### 3.2 Dominant Emotion Extraction
- For each song, selected the emotion with the highest score.
- **Output file**: `data/id_highest_gems.tsv` (columns: `song`, `emotion`)

### 3.3 User Emotion Profiles
- Merged emotion labels with interaction data.
- Computed per-user historical emotion distributions $P_u(e)$: normalized frequency of emotions in user listening history.

## 4. Recommendation Models

The following models were evaluated using Top-100 recommendations per user.

### 4.1 BPR (Bayesian Personalized Ranking)
- Pairwise ranking model trained on implicit feedback.
- Implemented using RecBole.
- **Output**: `outputs/02_base_recs/user_top100_BPR.tsv`

### 4.2 ItemKNN
- Item-based k-nearest-neighbors recommender.
- Implemented using RecBole.
- **Output**: `outputs/02_base_recs/user_top100_itemknn.tsv`

### 4.3 MostPop
- Non-personalized popularity baseline.
- Recommends the globally most frequently listened songs.
- **Output**: `outputs/02_base_recs/user_top100_mostpop.tsv`

### 4.4 Random
- Random baseline.
- Uniformly samples items.
- **Output**: `outputs/02_base_recs/user_top100_random.tsv`

## 5. Evaluation Metrics

### 5.1 Ranking Accuracy
- **nDCG@10**: Measures how well the top-10 recommendations match past user interactions.

### 5.2 Emotional Calibration (Top-10)
Calibration is computed by comparing:
- $P_u$: user’s historical emotion distribution
- $Q_u$: emotion distribution of the Top-10 recommendations

**Metrics**:
- **KL@10**: Kullback–Leibler divergence (directional)
- **JSD@10**: Jensen–Shannon divergence (symmetric, bounded)
- *Lower values indicate better emotional alignment.*

## 6. Baseline Evaluation
Computed nDCG@10, KL@10, and JSD@10 for all models.
- **Saved**:
    - `outputs/04_evaluation/evaluation_summary.csv` (mean & std per model)
    - `outputs/04_evaluation/calibration_all_models.csv` (per-user metrics)
- **Visualizations**: Boxplots for nDCG@10, KL@10, and JSD@10.

## 7. CaliTune-Style Post-hoc Calibration (Main Contribution)

### 7.1 Motivation
Initial heuristic re-ranking approaches did not guarantee improved calibration for higher λ values. To address this, the project adopts the greedy list-level re-ranking strategy from the CaliTune paper.

### 7.2 Method
- Applied post-hoc greedy re-ranking to BPR, ItemKNN, and MostPop recommendations.
- Built the recommendation list one item at a time.
- At each step, selected the item that maximizes:
  $$ (1 - \lambda) \cdot \text{relevance} - \lambda \cdot \text{JSD}(P_u, Q_L) $$
  - Relevance is approximated from the original ranking.
  - Calibration is computed explicitly at the list level.

### 7.3 Practical Optimizations
- Re-ranking restricted to **Top-50 candidates** per user (as in CaliTune).
- Used integer emotion indices instead of strings.
- Avoided expensive list operations in inner loops.

### 7.4 Outputs
Generated calibrated Top-10 recommendations for $\lambda \in \{0.0, 0.1, 0.3, 0.5, 0.7, 1.0\}$.
- **Output files**: `outputs/03_calibration/user_top10_<Model>_calitune_lambda_<λ>.tsv`

## 8. Lambda Sensitivity Analysis
Evaluated calibrated recommendations for each $\lambda$.
- **Expected behavior**:
    - $\lambda = 0 \to$ highest ranking accuracy
    - $\lambda \to 1 \to$ improved emotional calibration (lower KL/JSD)

## 9. Current Status
- ✅ Data preparation completed
- ✅ Baseline models evaluated
- ✅ Emotional calibration metrics implemented
- ✅ CaliTune-style greedy re-ranking applied to BPR, ItemKNN, and MostPop
