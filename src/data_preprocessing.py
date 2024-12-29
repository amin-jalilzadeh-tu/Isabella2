# src/data_preprocessing.py

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch

# Import your utility functions for cost and carbon
from utils.carbon import calculate_total_carbon
from utils.cost import calculate_total_cost

# Define input CSV file paths and column labels
TIME_HORIZONS = [2020, 2050, 2100]

TIME_HORIZON_LABEL = "time_horizon"
SIMULATION_ID = "Simulation ID"
WINDOWS_U_FACTOR = 'windows_U_Factor'
GROUND_FLOOR_THERMAL_RESISTANCE = "groundfloor_thermal_resistance"
EXT_WALLS_THERMAL_RESISTANCE = "ext_walls_thermal_resistance"
ROOF_THERMAL_RESISTANCE = "roof_thermal_resistance"
ANNUAL_ENERGY_CONSUMPTION = "annual_energy_consumption"
TOTAL_COST = "total_cost"
TOTAL_CARBON_EMISSION = "total_carbon_emission"
COMFORT_DAYS = "comfort_days"

ELECTRICITY_BUILDING = "Electricity:Building"
ELECTRICITY_FACILITY = "Electricity:Facility"
GAS_CONSUMPTION = "Gas Consumption"
INDOOR_TEMPERATURE = "Zone Mean Air Temperature"

def get_csv_file_path(time_horizon: int) -> str:
    file_name = f"{time_horizon}_merged_simulation_results.csv"
    return os.path.join("inputs", file_name)

def load_and_preprocess_data(output_archives: bool = True):
    """
    Loads CSV files for multiple time horizons, merges them,
    computes cost/carbon/comfort, returns final df_inputs, df_outputs,
    and also returns the train/val/test DataLoaders (plus scalers).
    """

    # Initialize dataframes for inputs and outputs
    df_inputs = pd.DataFrame(columns=[
        SIMULATION_ID, TIME_HORIZON_LABEL, WINDOWS_U_FACTOR,
        GROUND_FLOOR_THERMAL_RESISTANCE, EXT_WALLS_THERMAL_RESISTANCE,
        ROOF_THERMAL_RESISTANCE
    ])

    df_outputs = pd.DataFrame(columns=[
        SIMULATION_ID, ANNUAL_ENERGY_CONSUMPTION,
        TOTAL_COST, TOTAL_CARBON_EMISSION, COMFORT_DAYS
    ])

    for time_horizon in TIME_HORIZONS:
        csv_path = get_csv_file_path(time_horizon)
        if not os.path.isfile(csv_path):
            print(f"Warning: Missing CSV file: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # If 'Simulation ID' naming mismatch
        if 'Simulation ID' not in df.columns and 'simulation_id' in df.columns:
            df.rename(columns={'simulation_id': 'Simulation ID'}, inplace=True)

        date_columns = [
            col for col in df.columns
            if col.startswith(f"{str(time_horizon)}-")
        ]

        for simulation_id, group in df.groupby(SIMULATION_ID):
            # Collect inputs
            df_inputs = pd.concat(
                [df_inputs,
                 pd.DataFrame({
                     SIMULATION_ID: [int(simulation_id)],
                     TIME_HORIZON_LABEL: [time_horizon],
                     WINDOWS_U_FACTOR: [group[WINDOWS_U_FACTOR].iloc[0]],
                     GROUND_FLOOR_THERMAL_RESISTANCE: [
                         group[GROUND_FLOOR_THERMAL_RESISTANCE].iloc[0]
                     ],
                     EXT_WALLS_THERMAL_RESISTANCE: [
                         group[EXT_WALLS_THERMAL_RESISTANCE].iloc[0]
                     ],
                     ROOF_THERMAL_RESISTANCE: [
                         group[ROOF_THERMAL_RESISTANCE].iloc[0]
                     ]
                 })],
                ignore_index=True
            )

            # Extract consumption/temperature
            try:
                if 'index' in group.columns:
                    electricity_building = group[group['index'].str.contains(ELECTRICITY_BUILDING)].iloc[0]
                    electricity_facility = group[group['index'].str.contains(ELECTRICITY_FACILITY)].iloc[0]
                    gas_consumption = group[group['index'].str.contains(GAS_CONSUMPTION)].iloc[0]
                    indoor_temperature = group[group['index'].str.contains(INDOOR_TEMPERATURE)].iloc[0]
                else:
                    electricity_building = group[group.index.to_series().str.contains(ELECTRICITY_BUILDING)].iloc[0]
                    electricity_facility = group[group.index.to_series().str.contains(ELECTRICITY_FACILITY)].iloc[0]
                    gas_consumption = group[group.index.to_series().str.contains(GAS_CONSUMPTION)].iloc[0]
                    indoor_temperature = group[group.index.to_series().str.contains(INDOOR_TEMPERATURE)].iloc[0]

            except IndexError:
                print(f"Missing data for simulation ID {simulation_id} in time horizon {time_horizon}")
                continue

            # Compute annual energy consumption (example only uses gas here)
            annual_energy_consumption = (
                gas_consumption[date_columns].sum()
            ) / 1e9

            # Gather parameters
            window_U_factor = group[WINDOWS_U_FACTOR].iloc[0]
            groundfloor_thermal_resistance = group[GROUND_FLOOR_THERMAL_RESISTANCE].iloc[0]
            ext_walls_thermal_resistance = group[EXT_WALLS_THERMAL_RESISTANCE].iloc[0]
            roof_thermal_resistance = group[ROOF_THERMAL_RESISTANCE].iloc[0]

            # Calculate total cost, comfort, and carbon
            total_cost = calculate_total_cost(
                window_U_Factor=window_U_factor,
                groundfloor_thermal_resistance=groundfloor_thermal_resistance,
                ext_walls_thermal_resistance=ext_walls_thermal_resistance,
                roof_thermal_resistance=roof_thermal_resistance
            )

            comfort_days = min(
                len([item for item in indoor_temperature[date_columns].values if 17.5 < item < 24]),
                365
            )

            total_carbon_emission = calculate_total_carbon(
                window_U_Factor=window_U_factor,
                groundfloor_thermal_resistance=groundfloor_thermal_resistance,
                ext_walls_thermal_resistance=ext_walls_thermal_resistance,
                roof_thermal_resistance=roof_thermal_resistance
            )

            df_outputs = pd.concat(
                [df_outputs,
                 pd.DataFrame({
                     SIMULATION_ID: [int(simulation_id)],
                     ANNUAL_ENERGY_CONSUMPTION: [annual_energy_consumption],
                     TOTAL_COST: [total_cost],
                     TOTAL_CARBON_EMISSION: [total_carbon_emission],
                     COMFORT_DAYS: [comfort_days]
                 })],
                ignore_index=True
            )

    # Reset indexes for cleanliness
    df_inputs.reset_index(drop=True, inplace=True)
    df_outputs.reset_index(drop=True, inplace=True)

    if output_archives:
        # Save intermediate CSVs
        df_outputs.to_csv("ARCHIEVE/OUTPUT_GROUNDFLOOR.csv", index=False)
        df_inputs.to_csv("ARCHIEVE/INPUT_GROUNDFLOOR.csv", index=False)

    # Now define the columns for X and Y
    input_features = [
        TIME_HORIZON_LABEL,
        WINDOWS_U_FACTOR,
        GROUND_FLOOR_THERMAL_RESISTANCE,
        EXT_WALLS_THERMAL_RESISTANCE,
        ROOF_THERMAL_RESISTANCE
    ]
    target_features = [
        ANNUAL_ENERGY_CONSUMPTION,
        TOTAL_COST,
        TOTAL_CARBON_EMISSION,
        COMFORT_DAYS
    ]

    X_data = df_inputs[input_features].values
    Y_data = df_outputs[target_features].values

    # Normalize
    scaler_X = MinMaxScaler()
    X_data_normalized = scaler_X.fit_transform(X_data)

    # Normalize Y (one scaler per target)
    num_targets = Y_data.shape[1]
    scalers_Y = []
    Y_data_normalized_list = []
    for i in range(num_targets):
        scaler = MinMaxScaler()
        y = Y_data[:, i].reshape(-1, 1)
        y_normalized = scaler.fit_transform(y)
        scalers_Y.append(scaler)
        Y_data_normalized_list.append(y_normalized)

    Y_data_normalized = np.hstack(Y_data_normalized_list)

    # Train/Val/Test split
    train_inputs, temp_inputs, train_outputs, temp_outputs = train_test_split(
        X_data_normalized, Y_data_normalized, test_size=0.3, random_state=42
    )
    val_inputs, test_inputs, val_outputs, test_outputs = train_test_split(
        temp_inputs, temp_outputs, test_size=0.5, random_state=42
    )

    # Convert to Tensors
    train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
    train_outputs = torch.tensor(train_outputs, dtype=torch.float32)
    val_inputs = torch.tensor(val_inputs, dtype=torch.float32)
    val_outputs = torch.tensor(val_outputs, dtype=torch.float32)
    test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
    test_outputs = torch.tensor(test_outputs, dtype=torch.float32)

    # Create DataLoaders
    batch_size = 32
    train_dataset = TensorDataset(train_inputs, train_outputs)
    val_dataset = TensorDataset(val_inputs, val_outputs)
    test_dataset = TensorDataset(test_inputs, test_outputs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return {
        "df_inputs": df_inputs,
        "df_outputs": df_outputs,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "scaler_X": scaler_X,
        "scalers_Y": scalers_Y,
        "input_features": input_features,
        "target_features": target_features
    }
