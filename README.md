# Emotion-calibrated Music Recommendation

This repository contains the code for the paper Emotion-calibrated Music Recommendation, under review for UMAP'26.

The project analyzes a music catalog in terms of the GEMS [[1]](#1) emotion annotations provided with SRGNN-Emo [[2]](#2). It then applies calibration [[3]](#3) to the music recommendation, matching the distribution of emotions to the user's past listenings. The catalog used is extracted from the Music4All-Onion dataset [[4]](#4).

## Prerequisites

### Environment Setup

Create a conda environment and install the dependencies:

```bash
conda create -n emotion-recsys python=3.10
conda activate emotion-recsys
pip install -r requirements.txt
```

Key dependencies: `pandas`, `numpy`, `scipy`, `matplotlib`, `torch`, `recbole`.

### Data

Place the following files in the `data/` directory before running the pipeline:

| File | Description | Source |
|------|-------------|--------|
| `data/listening_history.csv` | User-song interaction logs (play counts) | [Zenodo](https://zenodo.org/records/18431594) |
| `data/id_gems.tsv` | Multi-dimensional GEM emotion scores per song | [Zenodo]() |

### Project Structure

```
├── preprocess_interactions.py                              # Step 1: Data preprocessing
├── scripts/helpers/
│   ├── convert_to_inter.py               # Step 1: Convert to RecBole format
│   └── convert_highest_gem.py            # Step 2: Extract dominant emotion per song
├── train_model.py                        # Step 3: Train RecBole models
├── generate_*_recommendations.py         # Step 3: Generate base recommendations
├── calibrate_*_greedy.py                 # Step 4: Greedy calibration re-ranking
├── evaluate_all_models.py                # Step 5: Base model evaluation
├── compare_calibration_study.py          # Step 5: Calibration trade-off analysis
├── evaluate_lambda_table.py              # Step 5: Lambda sensitivity table
├── calibration_utils.py                  # Shared calibration utilities
├── plot_emotion_distributions.py         # Visualization
├── plot_user_profile_distribution.py     # Visualization
├── recbole_configs/                      # RecBole model configurations
│   ├── Recbole_BPR_config_my_dataset.yaml
│   └── Recbole_ItemKNN_config_my_dataset.yaml
├── data/                                 # Input data (not tracked)
├── outputs/                              # Generated outputs (not tracked)
└── calibration/                          # RecBole working directory (not tracked)
```

## 1. Data Preparation

### 1.1 Raw Listening History
- **Input file**: `data/listening_history.csv`
- Contains user-song interaction logs (play counts). Can be downloaded [here](https://zenodo.org/records/18431594)
- **Data cleaning steps**:
    - Removed invalid or empty user/song entries.
    - Aggregated play counts per (user, song) pair.

### 1.2 Interaction Binarization
- Converts play counts into implicit feedback.
- **Workflow**:
    1. Keeps only (user, song) pairs with **>= 2 listens** to remove noise.
    2. Applies **5-core filtering** (iteratively removes users and songs with < 5 interactions).
- Assigns a fixed implicit feedback label (`label = 5`, compatible with RecBole).
- **Output file**: `outputs/01_preprocessing/interactions_binarized.csv`

**Run:**

```bash
python preprocess_interactions.py
```

This reads `data/listening_history.csv`, applies binarization and 5-core filtering, and writes the binarized interactions to `outputs/01_preprocessing/interactions_binarized.csv`.

### 1.3 Conversion to RecBole Format
- Converts interactions into RecBole-compatible `.inter` schema:
  ```
  user:token    item:token    label:float
  ```
- Enables training and inference with RecBole models.

**Run:**

```bash
python scripts/helpers/convert_to_inter.py
```

This converts `outputs/01_preprocessing/interactions_binarized.csv` into `calibration/data/my_dataset/my_dataset.inter`.

## 2. Emotion Metadata Processing

### 2.1 GEM Emotion Labels
- **Input file**: `data/id_gems.tsv` can be downloaded [here]()
- Contains multi-dimensional GEMS emotion annotations per song. 

### 2.2 Dominant Emotion Extraction
- For each song, selects the emotion with the highest score.
- **Output file**: `data/id_highest_gems.tsv` (columns: `id`, `highest_gem`)

**Run:**

```bash
python scripts/helpers/convert_highest_gem.py
```

This reads `data/id_gems.tsv`, picks the dominant emotion per song, and writes the result to `data/id_highest_gems.tsv`.

### 2.3 User Emotion Profiles
- Merges emotion labels with interaction data.
- Computes per-user historical emotion distributions $P_u$: normalized frequency of emotions in user listening history.
- This step is performed automatically by the calibration scripts (Section 6) via `calibration_utils.py`. No separate script needs to be run.

## 3. Recommendation Models

### 3.1 Training

Train the BPR [[5]](#5) and ItemKNN [[6]](#6) models using RecBole. Configuration files are in `recbole_configs/`. Trained models are saved to `saved/`.

**Run:**

```bash
python train_model.py BPR
python train_model.py ItemKNN
```

### 3.2 Generating Recommendations

Generate Top-100 recommendation lists per user for all four models:

```bash
python generate_bpr_recommendations.py
python generate_itemknn_recommendations.py
python generate_mostpop_recommendations.py
python generate_random_recommendations.py
```

| Model | Description | Output |
|-------|-------------|--------|
| BPR | Pairwise ranking on implicit feedback | `outputs/02_base_recs/user_top100_BPR.tsv` |
| ItemKNN | Item-based k-nearest-neighbors | `outputs/02_base_recs/user_top100_itemknn.tsv` |
| MostPop | Non-personalized popularity baseline | `outputs/02_base_recs/user_top100_mostpop.tsv` |
| Random | Uniform random item sampling | `outputs/02_base_recs/user_top100_random.tsv` |

> **Note:** MostPop loads the BPR checkpoint to access the dataset split. Random reads the interactions and BPR output files directly — neither requires its own trained model.

## 4. Evaluation Metrics

### 4.1 Ranking Accuracy
- **nDCG@10**: Measures how well the top-10 recommendations match past user interactions.

### 4.2 Emotional Calibration (Top-10)
Calibration is computed by comparing:
- $P_u$: user’s historical emotion distribution
- $R_u$: emotion distribution of the Top-10 recommendations

**Metrics**:
- **JSD@10**: Jensen–Shannon divergence  (*Lower values indicate better emotional alignment.*)
- **JSD@10**: Kullback-Leibler divergence  (*Lower values indicate better emotional alignment.*)
- **NDCG@10**: Normalized discounted cumulative gain (*Higher values indicate better accuracy.*)

## 5. Baseline Evaluation

Computes nDCG@10 and JSD@10 for all models. Also generates boxplot figures for each metric.

**Run:**

```bash
python evaluate_all_models.py
```

**Outputs:**
- `outputs/04_evaluation/evaluation_summary.csv` (mean & std per model)
- `outputs/04_evaluation/calibration_all_models.csv` (per-user KL and JSD)
- `outputs/04_evaluation/*_boxplot.png` (boxplots per metric)

## 6. Post-processing Calibration

### 6.1 Method
Applies post-hoc greedy re-ranking to BPR, ItemKNN, and MostPop recommendations. At each step, selected the item that maximizes:
  $$ (1 - \lambda) \cdot \text{relevance} - \lambda \cdot \text{JSD}(P_u, Q_L) $$
Relevance is approximated from the original ranking, or from the model's recommendation score Calibration is computed explicitly at the list level. Re-ranking restricted to **Top-100 candidates** per user.

### 6.2 Run

Apply greedy re-ranking to each model's Top-100 recommendations:

```bash
python calibrate_bpr_greedy.py
python calibrate_itemknn_greedy.py
python calibrate_mostpop_greedy.py
```

The `LAMBDAS` list at the top of each script controls which $\lambda$ values are evaluated. By default it is set to `[0.5]`. To run on all lambdas, change it to e.g. `[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`.

**Output files**: `outputs/03_calibration/user_top10_<model>_calibrated_model_lambda_<λ>.tsv`

## 7. Lambda Sensitivity Analysis

Evaluates calibrated recommendations for each $\lambda$, producing a summary table and trade-off plots.

**Run:**

```bash
python evaluate_lambda_table.py
python compare_calibration_study.py
```

**Outputs:**
- `outputs/04_evaluation/lambda_table_formatted.csv` (nDCG, KL, JSD per model and $\lambda$)
- `outputs/04_evaluation/comparison_study_results.csv` (detailed per-method results)
- `outputs/04_evaluation/comparison_<model>_tradeoff.png` (accuracy vs calibration trade-off plots)

> **Note:** Both scripts have a `LAMBDAS` list at the top that must match the $\lambda$ values used in the calibration step (Section 6).



## References
<a id="1">[1]</a> 
