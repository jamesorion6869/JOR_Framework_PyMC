import os
import warnings

# Aggressive fix for OpenMP warnings - MUST come before any other imports
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_THREADING_LAYER"] = "GNU" 

import pandas as pd
from jor_pymc import run_jor_pymc_safe

def main():
    # Suppress warnings in the main thread
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")

    input_csv = "jor_scores.csv"
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    df_original = pd.read_csv(input_csv)

    df_for_pymc = pd.DataFrame({
        'case_name': df_original['Case'],
        'C_score': df_original['C'],
        'E_score': df_original['E'],
        'P_score': df_original['P'],
        'flight_mod': df_original['Flight_Mod']
    })

    print("Starting vectorized PyMC sampling...")
    results_df = run_jor_pymc_safe(
        df_for_pymc,
        chains=4,
        cores=4,
        target_accept=0.95,
        draws=1000,
        tune=1000
    )

    df_merged = df_original.merge(results_df, left_on="Case", right_on="case_name", how="left")
    df_merged.drop(columns=['case_name'], inplace=True)

    decimal_cols = ['Posterior_Mean', 'CI_2.5%', 'CI_97.5%']
    for col in decimal_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].round(3)

    df_merged.to_csv(input_csv, index=False)
    print(f"Success! JOR CSV updated: {input_csv}")

if __name__ == "__main__":
    main()