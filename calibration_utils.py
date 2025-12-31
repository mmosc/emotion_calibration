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

def rerank_greedy_jsd(user_id, items, P_user_dist, song2emotion, emo2idx, num_emotions, lam, top_k):
    """
    Greedy re-ranking minimizing JSD divergence.
    """
    selected = []
    remaining = items.copy()
    
    q_counts = np.zeros(num_emotions, dtype=np.float64)
    n = len(items)
    rank_rel = {it: (n - r) / n for r, it in enumerate(items)}
    
    while len(selected) < top_k and remaining:
        best_item = None
        best_util = -np.inf
        best_emo_idx = None
        
        for it in remaining:
            emo = song2emotion.get(it)
            if emo not in emo2idx:
                continue
                
            eidx = emo2idx[emo]
            
            # Simulated Q distribution
            q_tmp = q_counts.copy()
            q_tmp[eidx] += 1.0
            q_dist = q_tmp / (len(selected) + 1)
            
            # Divergence (JSD)
            div = jensenshannon(P_user_dist + 1e-12, q_dist + 1e-12, base=2) ** 2
            
            rel = rank_rel[it]
            util = (1 - lam) * rel - lam * div
            
            if util > best_util:
                best_util = util
                best_item = it
                best_emo_idx = eidx
        
        if best_item is None:
            selected.extend(remaining[:(top_k - len(selected))])
            break
            
        selected.append(best_item)
        q_counts[best_emo_idx] += 1.0
        remaining.remove(best_item)
        
    return selected[:top_k]
