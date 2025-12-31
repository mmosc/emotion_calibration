import pandas as pd

# Load calibrated models table
try:
    df_models = pd.read_csv("outputs/04_evaluation/lambda_table_all_models.csv")
except FileNotFoundError:
    print("Warning: lambda_table_all_models.csv not found.")
    df_models = pd.DataFrame()

# Load Random summary
try:
    df_random = pd.read_csv("outputs/04_evaluation/random_evaluation_summary.csv")
except FileNotFoundError:
    print("Warning: random_evaluation_summary.csv not found.")
    df_random = pd.DataFrame()

if not df_models.empty and not df_random.empty:
    # Ensure same column order
    
    common_cols = [c for c in df_models.columns if c in df_random.columns]

    if not common_cols:
        
         print("Columns do not match exactly. Concatenating naively.")
    
    # Append Random as last row
    df_final = pd.concat([df_models, df_random], ignore_index=True)
else:
    df_final = pd.concat([df_models, df_random], ignore_index=True)

# Save final table
if not df_final.empty:
    df_final.to_csv("outputs/04_evaluation/final_results_table.csv", index=False)
    print("Saved outputs/04_evaluation/final_results_table.csv")
    print(df_final.tail())
else:
    print("No data to merge.")
