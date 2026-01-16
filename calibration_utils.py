import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import logging

def load_interactions_and_gems(interactions_path, gems_path):
    """
    Load interactions and gems data.
    """
    print("Loading data...")
    inter = pd.read_csv(interactions_path)
    inter.columns = ["user", "song", "label"]

    gems = pd.read_csv(gems_path, sep="\t")
    gems = gems.rename(columns={"id": "song", "highest_gem": "emotion"})
    
    song2emotion = dict(zip(gems["song"], gems["emotion"]))
    emotions = sorted(gems["emotion"].dropna().unique())
    
    return inter, gems, song2emotion, emotions

def build_emotion_distribution(inter, gems):
    """
    Build user emotion distribution P_u.
    """
    merged = inter.merge(gems, on="song", how="left")
    counts = merged.groupby(["user", "emotion"], observed=True).size().unstack(fill_value=0)
    
    # Normalize to probabilities
    P = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    print(f"Built emotion profiles for {len(P)} users.")
    return P

def rerank_linear(user_id, items, P_user, song2emotion, lam, top_k):
    """
    Linear re-ranking: score = (1-lambda)*rank_score + lambda*calibration_score
    """
    n = len(items)
    scored_items = []

    for rank, item in enumerate(items):
        base_score = (n - rank) / n
        emo = song2emotion.get(item)
        calib = P_user.get(emo, 0.0) if emo else 0.0
        
        new_score = (1 - lam) * base_score + lam * calib
        scored_items.append((item, new_score))
    
    scored_items.sort(key=lambda x: x[1], reverse=True)
    return [item for item, _ in scored_items[:top_k]]

def rerank_greedy_jsd(user_id, items, P_user_dist, song2emotion, emo2idx, num_emotions, lam, top_k, scores=None, score_type='rank'):
    """
    Greedy re-ranking minimizing JSD divergence.
    score_type: 'rank' uses (n-r)/n, 'model' uses raw scores (normalized to [0,1]).
    """
    selected = []
    remaining = list(range(len(items))) # Indices of items in the 'items' list
    
    q_counts = np.zeros(num_emotions, dtype=np.float64)
    n = len(items)
    
    # 1. Determine Relevance Scores
    if score_type == 'model' and scores is not None and len(scores) > 0:
        # Min-Max Normalization to [0, 1]
        s = np.array(scores, dtype=np.float64)
        min_v = s.min()
        max_v = s.max()
        if max_v > min_v:
            normalized_rel = (s - min_v) / (max_v - min_v)
        else:
            normalized_rel = np.ones_like(s) # Fallback if all scores are identical
        relevance_map = {i: normalized_rel[i] for i in range(len(items))}
    else:
        # 'rank': Fallback to rank-based scoring: (n-i)/n
        relevance_map = {i: (n - i) / n for i in range(len(items))}
    
    # 2. Greedy Selection Loop
    while len(selected) < top_k and remaining:
        best_idx = None
        best_util = -np.inf
        best_emo_idx = None
        
        for idx in remaining:
            it = items[idx]
            emo = song2emotion.get(it)
            if emo not in emo2idx:
                continue
                
            eidx = emo2idx[emo]
            
            # Simulated Q distribution
            q_tmp = q_counts.copy()
            q_tmp[eidx] += 1.0
            q_dist = q_tmp / (len(selected) + 1)
            
            # Divergence (JSD) - lower is better, so we subtract it
            div = jensenshannon(P_user_dist + 1e-12, q_dist + 1e-12, base=2) ** 2
            
            rel = relevance_map[idx]
            util = (1 - lam) * rel - lam * div
            
            if util > best_util:
                best_util = util
                best_idx = idx
                best_emo_idx = eidx
        
        if best_idx is None:
            # Fallback for remaining slots if items have no emotion data
            for idx in remaining:
                if len(selected) < top_k:
                    selected.append(items[idx])
            break
            
        selected.append(items[best_idx])
        q_counts[best_emo_idx] += 1.0
        remaining.remove(best_idx)
        
    return selected[:top_k]

