import pandas as pd
import numpy as np

# Define input and output file paths
input_csv_path = "C:\\Users\\Emra\\Desktop\\PR\\interactions_binarized.csv"
output_inter_path = "C:\\Users\\Emra\\Desktop\\PR\\CaliTune\\data\\my_dataset\\my_dataset.inter"

# Read the CSV file
df = pd.read_csv(input_csv_path)

# Rename columns to RecBole's expected format
# Define the header for the .inter file
# user:token, item:token, label:float
header = "user:token\titem:token\tlabel:float"

# Write the header and then the DataFrame to the .inter file
with open(output_inter_path, 'w') as f:
    f.write(header + '\n')
    df.to_csv(f, sep='\t', index=False, header=False)

print(f"Successfully converted {input_csv_path} to {output_inter_path}")
