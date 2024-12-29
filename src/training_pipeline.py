# src/training_pipeline.py

import torch
import joblib
import numpy as np

# Imports from your own modules
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
from .training_functions import (
    train_weighted_sum,
    train_mgda,
    train_uncertainty,
    train_cagrad,
    get_model_path,
    get_scaler_path
)
from .evaluation_functions import plot_loss_curves

import torch

def train_models_if_needed(
    X_data_normalized,
    scalers_Y,
    scaler_X,
    train_loader,
    val_loader,
    do_training=True,
    method_list=None,
    model_type_list=None,
    weights=None,
    num_epochs=100,
    learning_rate=1e-4,
    hidden_size=256
):
    """
    Orchestrates the training of multiple model types and methods if do_training=True;
    otherwise, it can skip training.

    :param X_data_normalized: np.ndarray, the normalized training features (used just for dimension reference).
    :param scalers_Y: list of fitted scalers for each target variable.
    :param scaler_X: the fitted scaler for inputs.
    :param train_loader: DataLoader for training data.
    :param val_loader: DataLoader for validation data.
    :param do_training: bool, if False, skip training (e.g., models are already trained).
    :param method_list: list of training methods to use (e.g., ['weighted_sum','mgda','uncertainty']).
    :param model_type_list: list of model architectures to train.
    :param weights: e.g., [1.0, 1.0, 1.0, 1.0] for Weighted Sum or None for other methods.
    :param num_epochs: number of epochs to train each model.
    :param learning_rate: learning rate for Adam optimizer.
    :param hidden_size: integer hidden dimension for simpler models.

    :return: None (models are saved to disk)
    """

    if not do_training:
        print("Skipping training since do_training=False.")
        return

    # Default values if not passed
    if method_list is None:
        method_list = ['weighted_sum', 'mgda', 'uncertainty']  # or add 'cagrad'

    if model_type_list is None:
        model_type_list = [
            'shared',
            'separate',
            'Ref_Based_Isa',
            'Data_Based_Isa',
            'More_Shared_Layer',
            'Few_Shared_Layers',
            'Deep_Balanced_Layer'
        ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = X_data_normalized.shape[1]  # e.g., 5

    for method in method_list:
        for model_type in model_type_list:
            print(f"\nTraining {model_type.capitalize()} Model with {method.replace('_', ' ').capitalize()} Method:")

            # 1) Instantiate the model
            if method == 'uncertainty':
                # Models for Uncertainty-based weighting
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
                    raise ValueError(f"Invalid model type: {model_type} for 'uncertainty' method.")

            elif method == 'cagrad':
                # Models for CAGrad
                if model_type == 'shared':
                    model = SharedMTLModel(input_size, hidden_size)
                elif model_type == 'separate':
                    model = SeparateMTLModel(input_size, hidden_size)
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
                    raise ValueError(f"Invalid model type: {model_type} for 'cagrad' method.")

            else:
                # Weighted_Sum or MGDA or default
                if model_type == 'shared':
                    model = SharedMTLModel(input_size, hidden_size)
                elif model_type == 'separate':
                    model = SeparateMTLModel(input_size, hidden_size)
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
                    raise ValueError(f"Invalid model type: {model_type} for method='{method}'.")

            # 2) Move model to device
            model.to(device)

            # 3) Train the model
            from .training_functions import (
                train_weighted_sum,
                train_mgda,
                train_uncertainty,
                train_cagrad
            )

            if method == 'weighted_sum':
                # Requires `weights`
                if weights is None:
                    raise ValueError("Weights must be provided for Weighted Sum method.")
                history = train_weighted_sum(
                    model, train_loader, val_loader, 
                    num_epochs, learning_rate, weights,
                    method=method, model_type=model_type
                )
            elif method == 'mgda':
                history = train_mgda(
                    model, train_loader, val_loader, 
                    num_epochs, learning_rate,
                    method=method, model_type=model_type
                )
            elif method == 'uncertainty':
                history = train_uncertainty(
                    model, train_loader, val_loader, 
                    num_epochs, learning_rate,
                    method=method, model_type=model_type
                )
            elif method == 'cagrad':
                history = train_cagrad(
                    model, train_loader, val_loader, 
                    num_epochs, learning_rate,
                    method=method, model_type=model_type
                )
            else:
                raise ValueError(f"Invalid method specified: {method}")

            # 4) Save the trained model
            model_path = get_model_path(method, model_type)
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

            # 5) Save the scalers
            scaler_X_path = get_scaler_path(method, model_type, scaler_type='X')
            joblib.dump(scaler_X, scaler_X_path)
            print(f"Input scaler saved to {scaler_X_path}")

            for i, scaler in enumerate(scalers_Y):
                scaler_Y_path = get_scaler_path(method, model_type, scaler_type=f'Y_{i}')
                joblib.dump(scaler, scaler_Y_path)
                print(f"Scaler for Y_{i} saved to {scaler_Y_path}")

            # 6) Plot the loss curves
            from .evaluation_functions import plot_loss_curves
            plot_loss_curves(history, method=method, model_type=model_type)
            print(f"Loss curves plotted for {model_type}_{method}")
