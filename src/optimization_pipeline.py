# src/optimization_pipeline.py

import numpy as np
import pandas as pd
import torch
import joblib

from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem

# If you keep them in training_functions or a similar file:
from .training_functions import get_scaler_path, get_model_path
# If you keep model selection in your inference code:
from .inference_pipeline import select_model


##########################################################################
# 11.1 User-Driven Multi-Objective Optimization (Pareto + MCDM)
##########################################################################
def user_driven_moo(predictions, df_inputs):
    """
    User-Driven Multi-Objective Optimization (Pareto + MCDM).
    The 'predictions' is a (num_samples, 4) array corresponding to:
      [Annual Energy, Retrofit Cost, CO2 Emission, Comfort Days].
    The 'df_inputs' is a DataFrame containing the design variables.

    Returns:
      df_pareto_11_1: DataFrame containing Pareto-optimal solutions (Approach 11.1).
    """
    # 1) Extract objectives
    annual_energy_consumption = predictions[:, 0]
    total_retrofit_cost       = predictions[:, 1]
    total_carbon_emission     = predictions[:, 2]
    comfort_days              = predictions[:, 3]

    # Convert Comfort Days to a "minimization" objective by negating
    negative_comfort_days = -comfort_days

    # 2) Combine into a DataFrame
    pareto_df = pd.DataFrame({
        'Time Horizon': df_inputs['time_horizon'],
        'Windows U-Factor': df_inputs['windows_U_Factor'],
        'Ground Floor Thermal Resistance': df_inputs['groundfloor_thermal_resistance'],
        'External Walls Thermal Resistance': df_inputs['ext_walls_thermal_resistance'],
        'Roof Thermal Resistance': df_inputs['roof_thermal_resistance'],
        'Annual Energy Consumption': annual_energy_consumption,
        'Total Retrofit Cost': total_retrofit_cost,
        'Total CO2 Emission': total_carbon_emission,
        'Negative Comfort Days': negative_comfort_days
    })

    # 3) Pareto Efficiency Function
    def is_pareto_efficient(costs, maximize=None, return_mask=True):
        """
        Determine which points are Pareto-efficient. 
        'costs' shape: (n_points, n_objectives).
        'maximize': list of bools, same length as #objectives,
                    indicating which objectives to maximize.
        """
        if maximize is not None:
            assert len(maximize) == costs.shape[1]
            costs = costs * np.array([-1 if m else 1 for m in maximize])

        num_points = costs.shape[0]
        is_efficient = np.ones(num_points, dtype=bool)
        for i in range(num_points):
            if is_efficient[i]:
                # A point i is dominated if there's another point j that is better or equal
                is_efficient &= ~np.all(costs <= costs[i], axis=1) | np.any(costs < costs[i], axis=1)
                is_efficient[i] = True
        if return_mask:
            return is_efficient
        else:
            return np.where(is_efficient)[0]

    # 4) Identify Pareto-Optimal Solutions
    objectives = pareto_df[['Annual Energy Consumption',
                            'Total Retrofit Cost',
                            'Total CO2 Emission',
                            'Negative Comfort Days']].values
    # We want to minimize the first three, maximize comfort => "True" for the 4th
    pareto_mask = is_pareto_efficient(objectives, maximize=[False, False, False, True])
    pareto_solutions = pareto_df[pareto_mask].reset_index(drop=True)

    # This is the final DataFrame for Approach 11.1
    df_pareto_11_1 = pareto_solutions.copy()
    return df_pareto_11_1


##########################################################################
# 11.2 Constraint-Based Multi-Objective Optimization with Pymoo
##########################################################################

# Global or user-defined bounds
ALLOWED_YEARS = np.array([2020, 2050, 2100])
MIN_WINDOW_U_FACTOR = 0.2
MAX_WINDOW_U_FACTOR = 2.0
MIN_GROUND_FLOOR_RESISTANCE = 0.5
MAX_GROUND_FLOOR_RESISTANCE = 5.0
MIN_EXT_WALLS_RESISTANCE = 0.5
MAX_EXT_WALLS_RESISTANCE = 5.0
MIN_ROOF_RESISTANCE = 0.5
MAX_ROOF_RESISTANCE = 5.0


def load_model_and_scalers(method, model_type, num_targets, input_size, hidden_size):
    """
    Utility to load a trained model + X-scaler + Y-scalers for the constraint-based optimization.
    """
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Load X-scaler
    scaler_X_path = get_scaler_path(method, model_type, scaler_type='X')
    scaler_X = joblib.load(scaler_X_path)

    # 2) Load Y-scalers
    scalers_Y = []
    for i in range(num_targets):
        scaler_path = get_scaler_path(method, model_type, f'Y_{i}')
        y_scaler = joblib.load(scaler_path)
        scalers_Y.append(y_scaler)

    # 3) Load model
    model = select_model(method, model_type, input_size, hidden_size)
    model_path = get_model_path(method, model_type)
    model.load_state_dict(torch.load(model_path, map_location=dev))
    model.to(dev)
    model.eval()

    return model, scaler_X, scalers_Y, dev


class BuildingOptimizationProblem(ElementwiseProblem):
    """
    A custom Problem subclass for pymoo that uses your loaded model to
    evaluate the 4 objectives:
      1) Annual Energy
      2) Retrofit Cost
      3) CO2 Emission
      4) Negative Comfort
    """
    def __init__(self, model, scaler_X, scalers_Y, device):
        n_var = 5  # [time_horizon, windows_U, groundfloor_R, ext_walls_R, roof_R]
        xl = np.array([
            ALLOWED_YEARS.min(),
            MIN_WINDOW_U_FACTOR,
            MIN_GROUND_FLOOR_RESISTANCE,
            MIN_EXT_WALLS_RESISTANCE,
            MIN_ROOF_RESISTANCE
        ])
        xu = np.array([
            ALLOWED_YEARS.max(),
            MAX_WINDOW_U_FACTOR,
            MAX_GROUND_FLOOR_RESISTANCE,
            MAX_EXT_WALLS_RESISTANCE,
            MAX_ROOF_RESISTANCE
        ])
        super().__init__(
            n_var=n_var,
            n_obj=4,
            n_constr=0,
            xl=xl,
            xu=xu,
            type_var=np.double
        )
        self.model = model
        self.scaler_X = scaler_X
        self.scalers_Y = scalers_Y
        self.device = device

    def _evaluate(self, x, out, *args, **kwargs):
        # Snap the time_horizon to the nearest of {2020,2050,2100}
        time_horizon = x[0]
        closest_year = ALLOWED_YEARS[np.argmin(np.abs(ALLOWED_YEARS - time_horizon))]
        x[0] = closest_year

        # Scale input
        x_scaled = self.scaler_X.transform([x])

        # Inference
        tensor_in = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor_in)
        if isinstance(outputs, (tuple, list)):
            outputs = torch.cat(outputs, dim=1)
        outputs_np = outputs.cpu().numpy()

        # Inverse transform
        outputs_list = []
        for i, scaler in enumerate(self.scalers_Y):
            y_norm = outputs_np[0, i].reshape(-1, 1)
            y_orig = scaler.inverse_transform(y_norm)
            outputs_list.append(y_orig[0,0])

        # 4 objectives: minimize the first 3, maximize comfort => negative comfort
        f1 = outputs_list[0]  # Annual energy
        f2 = outputs_list[1]  # Retrofit cost
        f3 = outputs_list[2]  # CO2
        f4 = -outputs_list[3] # Negative comfort
        out["F"] = [f1, f2, f3, f4]


def constraint_based_moo(method, model_type, input_size=5, hidden_size=256,
                         num_targets=4, n_generations=200, pop_size=100):
    """
    Runs a constraint-based optimization using NSGA2 from pymoo.
    Returns a DataFrame with the Pareto-optimal variables & objectives.
    """
    # 1) Load model & scalers
    model, scaler_X, scalers_Y, dev = load_model_and_scalers(
        method, model_type, num_targets, input_size, hidden_size
    )

    # 2) Create Problem
    problem = BuildingOptimizationProblem(model, scaler_X, scalers_Y, dev)

    # 3) Create Algorithm
    from pymoo.algorithms.moo.nsga2 import NSGA2
    algorithm = NSGA2(
        pop_size=pop_size,
        eliminate_duplicates=True
    )

    # 4) Solve
    from pymoo.optimize import minimize
    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_generations),
        seed=42,
        verbose=True
    )

    # 5) Extract
    pareto_objectives = res.F  # shape (n_solutions, 4)
    pareto_variables  = res.X  # shape (n_solutions, 5)

    obj_cols = [
        'Annual Energy Consumption',
        'Total Retrofit Cost',
        'Total CO2 Emission',
        'Negative Comfort Days'
    ]
    var_cols = [
        'Time Horizon',
        'Windows U-Factor',
        'Ground Floor Thermal Resistance',
        'External Walls Thermal Resistance',
        'Roof Thermal Resistance'
    ]
    df_obj = pd.DataFrame(pareto_objectives, columns=obj_cols)
    df_var = pd.DataFrame(pareto_variables, columns=var_cols)

    # Combine
    df_pareto_11_2 = pd.concat([df_var, df_obj], axis=1)
    return df_pareto_11_2
