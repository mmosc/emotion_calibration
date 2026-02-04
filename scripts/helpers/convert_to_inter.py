import pandas as pd
import os

# Define input and output file paths
input_csv_path = "outputs/01_preprocessing/interactions_binarized.csv"
output_inter_path = "calibration/data/my_dataset/my_dataset.inter"

# Read the CSV file
df = pd.read_csv(input_csv_path)

# Rename columns to match RecBole config field names
df = df.rename(columns={"user": "user", "song": "item", "label": "label"})

# RecBole .inter files use tab separation and specific headers
# format: field_name:field_type
# Types: token, float, etc.
# We'll use user:token, item:token, label:float

# Create the .inter dataframe with the correct header format
inter_df = df.copy()
inter_df.columns = ["user:token", "item:token", "label:float"]

# Save to .inter file
os.makedirs(os.path.dirname(output_inter_path), exist_ok=True)
inter_df.to_csv(output_inter_path, sep='\t', index=False)

print(f"Successfully converted {input_csv_path} to {output_inter_path}")
print(f"Columns: {inter_df.columns.tolist()}")
print(f"Head:\n{inter_df.head()}")
