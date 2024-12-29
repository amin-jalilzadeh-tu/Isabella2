import numpy as np
import pandas as pd

def is_pareto_efficient(costs, maximize=None, return_mask=True):
    """
    Identifies Pareto-efficient points.
    :param costs: A 2D array where each row represents a solution and each column an objective.
    :param maximize: A list of booleans indicating whether each objective should be maximized.
    :param return_mask: If True, returns a boolean mask; otherwise, returns indices of Pareto points.
    """
    if maximize is not None:
        assert len(maximize) == costs.shape[1]
        costs = costs * np.array([-1 if m else 1 for m in maximize])

    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = ~np.all(costs[is_efficient] <= c, axis=1) | np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient if return_mask else np.where(is_efficient)[0]

def user_driven_optimization(predictions, input_data, objective_weights=None):
    """
    Perform Pareto optimization for user-driven scenarios.
    :param predictions: Predicted values for objectives (NumPy array).
    :param input_data: Input data corresponding to the predictions (DataFrame).
    :param objective_weights: User-specified weights for each objective (optional).
    :return: Pareto-optimal solutions DataFrame.
    """
    # Extract objectives
    annual_energy_consumption = predictions[:, 0]
    total_retrofit_cost = predictions[:, 1]
    total_carbon_emission = predictions[:, 2]
    comfort_days = predictions[:, 3]

    # Convert 'Comfort Days' to a minimization objective
    negative_comfort_days = -comfort_days

    # Combine objectives into a DataFrame
    objectives_df = pd.DataFrame({
        'Annual Energy Consumption': annual_energy_consumption,
        'Total Retrofit Cost': total_retrofit_cost,
        'Total CO2 Emission': total_carbon_emission,
        'Negative Comfort Days': negative_comfort_days
    })

    # Calculate Pareto-efficient solutions
    objectives_array = objectives_df.values
    pareto_mask = is_pareto_efficient(objectives_array, maximize=[False, False, False, True])
    pareto_solutions = objectives_df[pareto_mask]

    # Combine with input data
    combined_df = pd.concat([input_data[pareto_mask].reset_index(drop=True), pareto_solutions.reset_index(drop=True)], axis=1)

    return combined_df
