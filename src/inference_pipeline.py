# src/inference_pipeline.py

import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# If your code references your own modules:
from .model_definitions import (
    SharedMTLModel,
    SeparateMTLModel,
    Ref_Based,
    Data_Based,
    More_Shared,
    Few_Shared,
    Deep_Balanced,
    SharedMTLModelWithUncertainty,
    SeparateMTLModelWithUncertainty
)
from .training_functions import get_model_path, get_scaler_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######################################################################
# 1) select_model
######################################################################
def select_model(method, model_type, input_size, hidden_size=256):
    """
    Instantiates the correct architecture based on method & model_type.
    Adjust hidden_size, dropout, etc. as needed.
    """
    if method == 'uncertainty':
        if model_type == 'shared':
            model = SharedMTLModelWithUncertainty(input_size, hidden_size, num_tasks=4)
        elif model_type == 'separate':
            model = SeparateMTLModelWithUncertainty(input_size, hidden_size, num_tasks=4)
        elif model_type == 'Ref_Based_Isa':
            model = Ref_Based(input_size=input_size, hidden_size1=128, hidden_size2=64, dropout_rate=0.5, num_tasks=4)
        elif model_type == 'Data_Based_Isa':
            model = Data_Based(
                input_size=input_size, hidden_size1=128, hidden_size2=64,
                shared_energy_emission_size=32, shared_comfort_size=32,
                dropout_rate=0.5, num_tasks=4
            )
        elif model_type == 'More_Shared_Layer':
            model = More_Shared(
                input_size=input_size, hidden_size1=256, hidden_size2=128, hidden_size3=64,
                dropout_rate=0.5, num_tasks=4
            )
        elif model_type == 'Few_Shared_Layers':
            model = Few_Shared(
                input_size=input_size, hidden_size_shared=128, dropout_rate=0.5, num_tasks=4
            )
        elif model_type == 'Deep_Balanced_Layer':
            model = Deep_Balanced(
                input_size=input_size, hidden_size1=200, hidden_size2=100, hidden_size3=50,
                dropout_rate=0.5, num_tasks=4
            )
        else:
            raise ValueError(f"Unsupported model_type '{model_type}' for method 'uncertainty'.")

    elif method == 'cagrad':
        if model_type == 'shared':
            model = SharedMTLModel(input_size=input_size, hidden_size=hidden_size)
        elif model_type == 'separate':
            model = SeparateMTLModel(input_size=input_size, hidden_size=hidden_size)
        elif model_type == 'Ref_Based_Isa':
            model = Ref_Based(input_size=input_size, hidden_size1=128, hidden_size2=64, dropout_rate=0.5, num_tasks=4)
        elif model_type == 'Data_Based_Isa':
            model = Data_Based(
                input_size=input_size, hidden_size1=128, hidden_size2=64,
                shared_energy_emission_size=32, shared_comfort_size=32,
                dropout_rate=0.5, num_tasks=4
            )
        elif model_type == 'More_Shared_Layer':
            model = More_Shared(
                input_size=input_size, hidden_size1=256, hidden_size2=128, hidden_size3=64,
                dropout_rate=0.5, num_tasks=4
            )
        elif model_type == 'Few_Shared_Layers':
            model = Few_Shared(
                input_size=input_size, hidden_size_shared=128, dropout_rate=0.5, num_tasks=4
            )
        elif model_type == 'Deep_Balanced_Layer':
            model = Deep_Balanced(
                input_size=input_size, hidden_size1=200, hidden_size2=100, hidden_size3=50,
                dropout_rate=0.5, num_tasks=4
            )
        else:
            raise ValueError(f"Unsupported model_type '{model_type}' for method 'cagrad'.")

    else:
        # weighted_sum, mgda, or default
        if model_type == 'shared':
            model = SharedMTLModel(input_size=input_size, hidden_size=hidden_size)
        elif model_type == 'separate':
            model = SeparateMTLModel(input_size=input_size, hidden_size=hidden_size)
        elif model_type == 'Ref_Based_Isa':
            model = Ref_Based(input_size=input_size, hidden_size1=128, hidden_size2=64, dropout_rate=0.5, num_tasks=4)
        elif model_type == 'Data_Based_Isa':
            model = Data_Based(
                input_size=input_size, hidden_size1=128, hidden_size2=64,
                shared_energy_emission_size=32, shared_comfort_size=32,
                dropout_rate=0.5, num_tasks=4
            )
        elif model_type == 'More_Shared_Layer':
            model = More_Shared(
                input_size=input_size, hidden_size1=256, hidden_size2=128, hidden_size3=64,
                dropout_rate=0.5, num_tasks=4
            )
        elif model_type == 'Few_Shared_Layers':
            model = Few_Shared(
                input_size=input_size, hidden_size_shared=128, dropout_rate=0.5, num_tasks=4
            )
        elif model_type == 'Deep_Balanced_Layer':
            model = Deep_Balanced(
                input_size=input_size, hidden_size1=200, hidden_size2=100, hidden_size3=50,
                dropout_rate=0.5, num_tasks=4
            )
        else:
            raise ValueError(f"Unsupported model_type '{model_type}' for method '{method}'.")

    return model


######################################################################
# 2) perform_inference
######################################################################
def perform_inference(
    method,
    model_type,
    input_features,
    num_targets,
    input_size,
    hidden_size,
    df_inputs,
    get_model_path_fn=None,
    get_scaler_path_fn=None
):
    """
    Loads the trained model & scalers, then infers on new data (df_inputs).
    Returns predictions on the original scale for each task.

    :param method: e.g., 'weighted_sum', 'mgda', 'uncertainty', 'cagrad'.
    :param model_type: e.g. 'shared', 'Ref_Based_Isa', etc.
    :param input_features: list of columns to feed into the model.
    :param num_targets: number of tasks (e.g. 4).
    :param input_size: integer dimension of model input.
    :param hidden_size: integer dimension of hidden layers.
    :param df_inputs: DataFrame containing new or user-provided data.
    :param get_model_path_fn: optional override for get_model_path (default in training_functions).
    :param get_scaler_path_fn: optional override for get_scaler_path.

    :return: A (N, num_targets) numpy array with predictions on the original scale.
    """

    if get_model_path_fn is None:
        get_model_path_fn = get_model_path
    if get_scaler_path_fn is None:
        get_scaler_path_fn = get_scaler_path

    # 1) Load scalers
    scaler_X_path = get_scaler_path_fn(method, model_type, scaler_type='X')
    if not os.path.exists(scaler_X_path):
        raise FileNotFoundError(f"Scaler for X not found at path: {scaler_X_path}")
    scaler_X = joblib.load(scaler_X_path)

    scalers_Y = []
    for i in range(num_targets):
        scaler_Y_path = get_scaler_path_fn(method, model_type, scaler_type=f'Y_{i}')
        if not os.path.exists(scaler_Y_path):
            raise FileNotFoundError(f"Scaler for Y_{i} not found at path: {scaler_Y_path}")
        y_scaler = joblib.load(scaler_Y_path)
        scalers_Y.append(y_scaler)

    # 2) Select and instantiate the correct model
    model = select_model(method, model_type, input_size, hidden_size)

    # 3) Load the trained model weights
    model_path = get_model_path_fn(method, model_type)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at path: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 4) Prepare the input data
    X_new = df_inputs[input_features].values
    X_new_scaled = scaler_X.transform(X_new)
    X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)

    # 5) Make predictions
    with torch.no_grad():
        outputs = model(X_new_tensor)

    # 6) Combine outputs if multiple tasks
    if isinstance(outputs, (tuple, list)):
        predictions_normalized = torch.cat(outputs, dim=1).cpu().numpy()
    else:
        predictions_normalized = outputs.cpu().numpy()

    # 7) Inverse-transform each target
    predictions_original_scale_list = []
    for i in range(num_targets):
        y_pred_norm = predictions_normalized[:, i].reshape(-1, 1)
        y_scaler = scalers_Y[i]
        y_pred_original = y_scaler.inverse_transform(y_pred_norm)
        predictions_original_scale_list.append(y_pred_original)

    predictions_original_scale = np.hstack(predictions_original_scale_list)

    print("Predictions on original scale:\n", predictions_original_scale)
    return predictions_original_scale
