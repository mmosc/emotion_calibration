import pandas as pd

# Load data
inter = pd.read_csv("interactions_binarized.csv")
gems = pd.read_csv("id_highest_gems.tsv", sep="\t")


gems = gems.rename(columns={"id": "song", "highest_gem": "emotion"})

# Compute dataset statistics
stats = {
    "Unique users": inter['user'].nunique(),
    "Unique items": inter['song'].nunique(),
    "Total interactions": len(inter),
    "Avg interactions per user": len(inter) / inter['user'].nunique(),
    "Emotion categories": gems['emotion'].nunique()
}

df_stats = pd.DataFrame(stats, index=[0])

print(df_stats)
df_stats.to_csv("dataset_summary.csv", index=False)

