import numpy as np
import pandas as pd
import torch
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem

class BuildingOptimizationProblem(ElementwiseProblem):
    """
    Custom optimization problem for constraint-based retrofitting scenarios.
    """
    def __init__(self, model, scaler_X, scalers_Y, device, input_bounds, allowed_years):
        self.model = model
        self.scaler_X = scaler_X
        self.scalers_Y = scalers_Y
        self.device = device
        self.allowed_years = allowed_years

        super().__init__(
            n_var=len(input_bounds),
            n_obj=4,
            n_constr=0,
            xl=np.array([b[0] for b in input_bounds]),
            xu=np.array([b[1] for b in input_bounds])
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # Snap year to nearest allowed value
        x[0] = self.allowed_years[np.argmin(np.abs(self.allowed_years - x[0]))]

        # Scale inputs and run the model
        input_scaled = self.scaler_X.transform([x])
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs_normalized = self.model(input_tensor)

        if isinstance(outputs_normalized, (tuple, list)):
            outputs_normalized = torch.cat(outputs_normalized, dim=1)
        outputs_normalized = outputs_normalized.cpu().numpy()

        # Inverse transform the outputs
        outputs_original = []
        for i, scaler in enumerate(self.scalers_Y):
            outputs_original.append(scaler.inverse_transform(outputs_normalized[:, i].reshape(-1, 1))[0, 0])

        # Set objectives
        f1, f2, f3, f4 = outputs_original
        out["F"] = [f1, f2, f3, -f4]  # Minimize energy, cost, CO2; maximize comfort

def constraint_based_optimization(model, scaler_X, scalers_Y, device, input_bounds, allowed_years, pop_size=100, n_gen=200):
    """
    Perform constraint-based Pareto optimization using NSGA-II.
    :return: Pareto-optimal solutions.
    """
    problem = BuildingOptimizationProblem(model, scaler_X, scalers_Y, device, input_bounds, allowed_years)

    algorithm = NSGA2(
        pop_size=pop_size,
        eliminate_duplicates=True
    )

    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        seed=42,
        verbose=True
    )

    pareto_solutions = pd.DataFrame(res.X, columns=[
        'Time Horizon', 'Windows U-Factor', 'Ground Floor Resistance', 'Ext Walls Resistance', 'Roof Resistance'
    ])
    pareto_objectives = pd.DataFrame(res.F, columns=[
        'Annual Energy Consumption', 'Total Retrofit Cost', 'Total CO2 Emission', 'Negative Comfort Days'
    ])

    return pd.concat([pareto_solutions, pareto_objectives], axis=1)
