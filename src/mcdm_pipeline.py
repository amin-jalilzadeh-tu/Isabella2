# src/mcdm_pipeline.py

import numpy as np
import pandas as pd

from pymoo.decomposition.asf import ASF
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints

def perform_mcdm(pareto_df, approach='11_1'):
    """
    Performs Multi-Criteria Decision Making (MCDM) on the given Pareto front data.
    The DataFrame 'pareto_df' must contain these columns:
      - 'Annual Energy Consumption'
      - 'Total CO2 Emission'
      - 'Total Retrofit Cost'
      - 'Negative Comfort Days'

    :param pareto_df: DataFrame of Pareto solutions (Approach 11.1 or 11.2)
    :param approach: '11_1' or '11_2', controlling the weights used in Weighted Score
    :return: (pareto_df_sorted, best_asf, best_solution_pseudo, tradeoff_solutions)
    """
    # Make a copy to avoid altering the original
    df_local = pareto_df.copy()

    # Extract the relevant objective columns as a NumPy array
    # Order: [Annual Energy, Total CO2, Cost, Negative Comfort]
    F = df_local[['Annual Energy Consumption',
                  'Total CO2 Emission',
                  'Total Retrofit Cost',
                  'Negative Comfort Days']].values

    # Normalize the objectives
    # Remember, 'Negative Comfort Days' is already a minimization objective
    F_min = F.min(axis=0)
    F_max = F.max(axis=0)
    F_normalized = (F - F_min) / (F_max - F_min)

    # Define weights for each approach
    if approach == '11_1':
        weights_array = np.array([0.3, 0.3, 0.2, 0.2])  # Example weights
    elif approach == '11_2':
        weights_array = np.array([0.25, 0.25, 0.25, 0.25])
    else:
        raise ValueError("Approach must be either '11_1' or '11_2'.")

    # Weighted Score for each solution
    weighted_scores = np.dot(F_normalized, weights_array)
    df_local['Weighted_Score'] = weighted_scores

    # Sort ascending by Weighted Score
    df_local_sorted = df_local.sort_values(by='Weighted_Score', ascending=True).reset_index(drop=True)

    # ASF (Compromise Programming) using pymoo
    decomp = ASF()
    idx_asf = decomp.do(F, weights_array).argmin()
    best_solution_asf = df_local.iloc[idx_asf]

    # Pseudo-Weights
    pseudo_weights_method = PseudoWeights(weights_array)
    idx_pseudo, _ = pseudo_weights_method.do(F, return_pseudo_weights=True)
    best_solution_pseudo = df_local.iloc[idx_pseudo]

    # High Trade-off Points
    tradeoff_method = HighTradeoffPoints()
    idx_tradeoff = tradeoff_method.do(F)
    tradeoff_solutions = df_local.iloc[idx_tradeoff]

    # Print results (optional, you can remove or modify as needed)
    print(f"\n=== MCDM for Approach {approach} ===")
    print("Best Solution According to ASF (Compromise):")
    print(best_solution_asf)

    print("\nBest Solution According to Pseudo-Weights:")
    print(best_solution_pseudo)

    print("\nHigh Trade-off Solutions:")
    print(tradeoff_solutions)

    return df_local_sorted, best_solution_asf, best_solution_pseudo, tradeoff_solutions


def example_mcdm_flow(df_pareto_11_1, df_pareto_11_2):
    """
    Example function that shows how to apply MCDM for 
    Approach 11.1 (User-Driven) and 11.2 (Constraint-Based).
    """
    print("\nMCDM Results for Approach 11.1 (User-Driven Optimization):")
    df_11_1_mcdm, best_asf_11_1, best_pseudo_11_1, tradeoff_11_1 = perform_mcdm(df_pareto_11_1, approach='11_1')

    print("\nMCDM Results for Approach 11.2 (Constraint-Based Optimization):")
    df_11_2_mcdm, best_asf_11_2, best_pseudo_11_2, tradeoff_11_2 = perform_mcdm(df_pareto_11_2, approach='11_2')

    # Return them if needed
    return (df_11_1_mcdm, best_asf_11_1, best_pseudo_11_1, tradeoff_11_1,
            df_11_2_mcdm, best_asf_11_2, best_pseudo_11_2, tradeoff_11_2)
