import pandas as pd

# Load the tsv file
df = pd.read_csv('C:\\Users\\Emra\\Desktop\\PR\\CaliTune\\id_gems.tsv\\id_gems.tsv', sep='\t')

# Set the 'id' column as the index
df.set_index('id', inplace=True)

# Find the column with the maximum value for each row
df['highest_gem'] = df.idxmax(axis=1)

# Reset the index to get the 'id' column back
df.reset_index(inplace=True)

# Keeping only the 'id' and 'highest_gem' columns
result_df = df[['id', 'highest_gem']]

# Save the result to a new tsv file
result_df.to_csv('C:\\Users\\Emra\\Desktop\\PR2\\id_highest_gems.tsv', sep='\t', index=False)
