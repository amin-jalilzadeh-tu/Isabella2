# src/evaluation_functions.py

import os
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# scikit-learn imports
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn

# If your code references the training functions (MGDA, Weighted Sum, etc.) for cross_validation or ablation_study:
from .training_functions import (
    train_weighted_sum,
    train_mgda,
    train_uncertainty,
    train_cagrad
)


##########################################################################
# 7.1 Evaluation Function with Inverse Transformation
##########################################################################
def evaluate_model_inverse_transform(model, data_loader, scalers_Y, method='evaluation', model_type='shared'):
    """
    Evaluates the model and computes evaluation metrics (MAE, MSE, RMSE, R²),
    handling inverse scaling for each task output.
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_targets = []
    all_predictions = []
    num_tasks = len(scalers_Y)

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # outputs should be a tuple/list of shape = (batch_size, 1) per task
            preds = torch.cat(outputs, dim=1)  # Shape: [batch_size, num_tasks]
            all_predictions.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Convert to numpy arrays
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    # Inverse transform
    all_targets_real_list = []
    all_predictions_real_list = []
    for i in range(num_tasks):
        y_true = all_targets[:, i].reshape(-1, 1)
        y_pred = all_predictions[:, i].reshape(-1, 1)
        scaler = scalers_Y[i]
        y_true_real = scaler.inverse_transform(y_true)
        y_pred_real = scaler.inverse_transform(y_pred)
        all_targets_real_list.append(y_true_real)
        all_predictions_real_list.append(y_pred_real)

    all_targets_real = np.hstack(all_targets_real_list)
    all_predictions_real = np.hstack(all_predictions_real_list)

    # Evaluate metrics
    task_names = ['Energy', 'Cost', 'Emission', 'Comfort']
    metrics = {
        'Task': task_names,
        'MAE': [],
        'MSE': [],
        'RMSE': [],
        'R2': []
    }

    for i in range(num_tasks):
        mse = mean_squared_error(all_targets_real[:, i], all_predictions_real[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets_real[:, i], all_predictions_real[:, i])
        r2 = r2_score(all_targets_real[:, i], all_predictions_real[:, i])

        metrics['MAE'].append(mae)
        metrics['MSE'].append(mse)
        metrics['RMSE'].append(rmse)
        metrics['R2'].append(r2)

    metrics_df = pd.DataFrame(metrics).set_index('Task')
    print(metrics_df)

    return metrics_df, all_targets_real, all_predictions_real


##########################################################################
# 7.2 Evaluation Function for Percentage Error
##########################################################################
def evaluate_model_percentage_error(model, data_loader, scalers_Y,
                                    method='evaluation', model_type='shared',
                                    save_plots=True, clip_outliers=True):
    """
    Computes percentage error between predictions and ground truth (inverse scaled).
    Optionally plots error distributions.
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_targets = []
    all_predictions = []
    num_tasks = len(scalers_Y)

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.cat(outputs, dim=1)  # [batch_size, num_tasks]
            all_predictions.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    # Inverse transform
    all_targets_real_list = []
    all_predictions_real_list = []
    for i in range(num_tasks):
        y_true = all_targets[:, i].reshape(-1, 1)
        y_pred = all_predictions[:, i].reshape(-1, 1)
        scaler = scalers_Y[i]
        y_true_real = scaler.inverse_transform(y_true)
        y_pred_real = scaler.inverse_transform(y_pred)
        all_targets_real_list.append(y_true_real)
        all_predictions_real_list.append(y_pred_real)

    all_targets_real = np.hstack(all_targets_real_list)
    all_predictions_real = np.hstack(all_predictions_real_list)

    # Percentage Error
    epsilon = 1e-6
    percentage_error = 100 * (all_predictions_real - all_targets_real) / (np.abs(all_targets_real) + epsilon)

    # Plot histograms if requested
    if save_plots:
        task_names = ['Energy', 'Cost', 'Emission', 'Comfort']
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        for i in range(num_tasks):
            # optionally clip outliers
            error_data = np.clip(percentage_error[:, i], -500, 500) if clip_outliers else percentage_error[:, i]
            axes[i].hist(error_data, bins=50, alpha=0.7, color='g', edgecolor='black')
            axes[i].set_title(f'Pct Error - {task_names[i]}\n({method.capitalize()} - {model_type.capitalize()})')
            axes[i].set_xlabel('Percentage Error (%)')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    return percentage_error, all_targets_real, all_predictions_real


##########################################################################
# 7.3 Function to Rank Models
##########################################################################
def rank_models(evaluation_dict, weights=None):
    """
    Ranks models based on evaluation metrics, including real and normalized metrics.
    evaluation_dict: { model_name: metrics_df }
    weights: e.g. {'MAE': 0.3, 'RMSE': 0.3, 'R2': 0.4}
    """
    if weights is None:
        weights = {'MAE': 1, 'RMSE': 1, 'R2': 1}

    model_scores = []
    metric_values = {'MAE': [], 'RMSE': [], 'R2': []}

    # Collect all metric values across models
    for model, metrics_df in evaluation_dict.items():
        if metrics_df is None or metrics_df.empty:
            print(f"Metrics for model '{model}' are missing.")
            continue
        for metric in metric_values.keys():
            if metric in metrics_df.columns:
                value = metrics_df[metric].mean()
                metric_values[metric].append(value)
            else:
                print(f"Warning: Metric '{metric}' is missing for model '{model}'.")

    metric_min = {m: min(v) for m, v in metric_values.items() if v}
    metric_max = {m: max(v) for m, v in metric_values.items() if v}

    def normalize(metric, value):
        # For MAE & RMSE, lower is better -> invert
        # For R2, higher is better
        if metric in ['MAE', 'RMSE']:
            if metric_max[metric] - metric_min[metric] == 0:
                return 1
            return (metric_max[metric] - value) / (metric_max[metric] - metric_min[metric])
        elif metric == 'R2':
            if metric_max[metric] - metric_min[metric] == 0:
                return 1
            return (value - metric_min[metric]) / (metric_max[metric] - metric_min[metric])
        else:
            return 0

    # Compute composite scores
    for model, metrics_df in evaluation_dict.items():
        if metrics_df is None or metrics_df.empty:
            continue
        missing_metrics = [m for m in metric_values.keys() if m not in metrics_df.columns]
        if missing_metrics:
            print(f"Skipping model '{model}' due to missing metrics: {missing_metrics}")
            continue
        avg_mae = metrics_df['MAE'].mean()
        avg_rmse = metrics_df['RMSE'].mean()
        avg_r2 = metrics_df['R2'].mean()

        norm_mae = normalize('MAE', avg_mae)
        norm_rmse = normalize('RMSE', avg_rmse)
        norm_r2 = normalize('R2', avg_r2)

        composite_score = (
            weights.get('MAE', 1)*norm_mae +
            weights.get('RMSE', 1)*norm_rmse +
            weights.get('R2', 1)*norm_r2
        )

        model_scores.append({
            'Model': model,
            'MAE': avg_mae,
            'Norm_MAE': norm_mae,
            'RMSE': avg_rmse,
            'Norm_RMSE': norm_rmse,
            'R2': avg_r2,
            'Norm_R2': norm_r2,
            'Composite_Score': composite_score
        })

    if not model_scores:
        print("No models to rank or incomplete metrics.")
        return pd.DataFrame()

    ranked_df = pd.DataFrame(model_scores)
    ranked_df = ranked_df.sort_values(by='Composite_Score', ascending=False).reset_index(drop=True)
    columns_order = ['Model', 'MAE', 'Norm_MAE', 'RMSE', 'Norm_RMSE', 'R2', 'Norm_R2', 'Composite_Score']
    ranked_df = ranked_df[columns_order]
    return ranked_df


##########################################################################
# 7.4 Function to Visualize Ranked Models
##########################################################################
def visualize_ranked_models(ranked_df):
    """
    Visualizes the ranking of models based on their composite scores, as well as individual metrics.
    """
    if ranked_df.empty:
        print("Ranked DataFrame is empty. Nothing to display.")
        return

    sns.set(style="whitegrid")

    # Plot Composite Scores
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Composite_Score', y='Model', data=ranked_df, palette='viridis')
    plt.title('Model Rankings Based on Composite Scores')
    plt.xlabel('Composite Score')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.show()

    # Plot individual metrics
    metrics = ['MAE', 'RMSE', 'R2']
    for metric in metrics:
        if metric not in ranked_df.columns:
            print(f"Metric '{metric}' not found in ranked_df. Skipping its plot.")
            continue
        plt.figure(figsize=(10, 6))
        sns.barplot(x=metric, y='Model', data=ranked_df, palette='magma')
        plt.title(f'Model Rankings Based on {metric}')
        plt.xlabel(metric)
        plt.ylabel('Model')
        plt.tight_layout()
        plt.show()


##########################################################################
# 7.5 Function to Plot Loss Curves
##########################################################################
def plot_loss_curves(history, method, model_type, num_tasks=4):
    """
    Plots the training/validation total loss and per-task losses over epochs.
    """
    epochs = range(1, len(history['train_total_loss']) + 1)

    plt.figure(figsize=(14, 6))

    # Total Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_total_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_total_loss'], 'r-', label='Validation Loss')
    plt.title(f'Total Loss ({method.capitalize()} - {model_type.capitalize()})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Task-specific Losses
    plt.subplot(1, 2, 2)
    for i in range(num_tasks):
        train_task_loss = history['train_task_losses'][i][:len(epochs)]
        val_task_loss = history['val_task_losses'][i][:len(epochs)]
        plt.plot(epochs, train_task_loss, label=f'Train Task {i+1} Loss')
        plt.plot(epochs, val_task_loss, linestyle='--', label=f'Val Task {i+1} Loss')
    plt.title(f'Task Losses ({method.capitalize()} - {model_type.capitalize()})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


##########################################################################
# 7.6 Function to Plot Actual vs Predicted Values
##########################################################################
def plot_actual_vs_predicted(targets, predictions, method, model_type, num_tasks=4):
    """
    Plots actual vs. predicted for each of the 4 tasks in a 2x2 grid.
    """
    task_names = ['Energy', 'Cost', 'Emission', 'Comfort']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i in range(num_tasks):
        axes[i].scatter(targets[:, i], predictions[:, i], alpha=0.5)
        min_val = min(targets[:, i].min(), predictions[:, i].min())
        max_val = max(targets[:, i].max(), predictions[:, i].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[i].set_xlabel('Actual')
        axes[i].set_ylabel('Predicted')
        axes[i].set_title(
            f'Actual vs Predicted - {task_names[i]} Task\n({method.capitalize()} - {model_type.capitalize()})'
        )
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()


##########################################################################
# 7.7 Cross-Validation Evaluation Function
##########################################################################
def cross_validate_model(model_class, X, Y, scalers_Y,
                         k=5, num_epochs=100, learning_rate=1e-4, 
                         weights=None, method='weighted_sum', model_type='shared'):
    """
    k-fold cross validation. Trains your model_class on k folds, returning metrics.
    """
    from .training_functions import (
        train_weighted_sum,
        train_mgda,
        train_uncertainty,
        train_cagrad
    )

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold = 1
    cv_metrics = []

    for train_index, val_index in kf.split(X):
        print(f"\nStarting Fold {fold}/{k}")
        X_train_cv, X_val_cv = X[train_index], X[val_index]
        Y_train_cv, Y_val_cv = Y[train_index], Y[val_index]

        # Scale data
        scaler_X_cv = MinMaxScaler()
        X_train_cv_scaled = scaler_X_cv.fit_transform(X_train_cv)
        X_val_cv_scaled = scaler_X_cv.transform(X_val_cv)

        # Scale Y for each task
        scalers_Y_cv = []
        Y_train_cv_norm_list = []
        Y_val_cv_norm_list = []
        num_tasks = Y.shape[1]

        for i in range(num_tasks):
            scaler = MinMaxScaler()
            y_train = Y_train_cv[:, i].reshape(-1, 1)
            y_val = Y_val_cv[:, i].reshape(-1, 1)
            y_train_scaled = scaler.fit_transform(y_train)
            y_val_scaled = scaler.transform(y_val)
            scalers_Y_cv.append(scaler)
            Y_train_cv_norm_list.append(y_train_scaled)
            Y_val_cv_norm_list.append(y_val_scaled)

        Y_train_cv_normalized = np.hstack(Y_train_cv_norm_list)
        Y_val_cv_normalized = np.hstack(Y_val_cv_norm_list)

        # Convert to tensors
        train_inputs_cv = torch.tensor(X_train_cv_scaled, dtype=torch.float32)
        train_outputs_cv = torch.tensor(Y_train_cv_normalized, dtype=torch.float32)
        val_inputs_cv = torch.tensor(X_val_cv_scaled, dtype=torch.float32)
        val_outputs_cv = torch.tensor(Y_val_cv_normalized, dtype=torch.float32)

        # Dataloaders
        train_dataset_cv = TensorDataset(train_inputs_cv, train_outputs_cv)
        val_dataset_cv = TensorDataset(val_inputs_cv, val_outputs_cv)
        train_loader_cv = DataLoader(train_dataset_cv, batch_size=32, shuffle=True)
        val_loader_cv = DataLoader(val_dataset_cv, batch_size=32)

        # Initialize model
        model = model_class(input_size=X.shape[1], hidden_size=256)  # adjust if needed

        # Train
        if method == 'weighted_sum':
            _ = train_weighted_sum(
                model, train_loader_cv, val_loader_cv, 
                num_epochs, learning_rate, weights,
                method=method, model_type=model_type
            )
        elif method == 'mgda':
            _ = train_mgda(
                model, train_loader_cv, val_loader_cv, 
                num_epochs, learning_rate, 
                method=method, model_type=model_type
            )
        elif method == 'uncertainty':
            _ = train_uncertainty(
                model, train_loader_cv, val_loader_cv, 
                num_epochs, learning_rate, 
                method=method, model_type=model_type
            )
        elif method == 'cagrad':
            _ = train_cagrad(
                model, train_loader_cv, val_loader_cv, 
                num_epochs, learning_rate, 
                method=method, model_type=model_type
            )
        else:
            raise ValueError("Invalid method for cross_validation")

        # Evaluate
        metrics_df, targets_real, preds_real = evaluate_model_inverse_transform(
            model, val_loader_cv, scalers_Y_cv, method=method, model_type=model_type
        )

        cv_metrics.append(metrics_df)
        fold += 1

    # Aggregate
    avg_metrics = cv_metrics[0].copy()
    for metric in ['MAE', 'MSE', 'RMSE', 'R2']:
        avg_metrics[metric] = [df[metric].mean() for df in cv_metrics]

    avg_metrics['Task'] = ['Energy', 'Cost', 'Emission', 'Comfort']
    avg_metrics = avg_metrics.set_index('Task')
    print("\nCross-Validation Average Metrics:")
    print(avg_metrics)

    return cv_metrics, avg_metrics


##########################################################################
# 7.8 Feature Importance with SHAP
##########################################################################
class TaskSpecificModel(nn.Module):
    """
    A lightweight wrapper that extracts one task output from a multi-task model
    so that SHAP can analyze them individually.
    """
    def __init__(self, original_model, task_index):
        super(TaskSpecificModel, self).__init__()
        self.original_model = original_model
        self.task_index = task_index

    def forward(self, x):
        outputs = self.original_model(x)
        # outputs might be a tuple of T tasks
        if isinstance(outputs, (tuple, list)):
            task_output = outputs[self.task_index]
        else:
            # single-output model, or fallback
            task_output = outputs
        if isinstance(task_output, (tuple, list)):
            # in case it's another nested tuple
            task_output = task_output[0]
        return task_output


def compute_shap_values(model, data_loader, scalers_X, feature_names,
                        method='evaluation', model_type='shared',
                        num_samples=100):
    """
    Uses SHAP to compute feature attributions for each task in a multi-task model.
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    X_sample = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            X_sample.append(inputs)
            if len(X_sample) * inputs.size(0) >= num_samples:
                break
    X_sample = torch.cat(X_sample, dim=0)[:num_samples].cpu().numpy()
    X_sample_scaled = X_sample  # Assuming data_loader outputs are already scaled

    # typical 4 tasks
    task_names = ['Energy', 'Cost', 'Emission', 'Comfort']
    assert len(feature_names) == X_sample_scaled.shape[1], (
        "Feature names do not match input dimensions!"
    )

    shap_values_dict = {}

    for i, task in enumerate(task_names):
        print(f"\nComputing SHAP values for {task} Task")
        # Wrap the model to output only that one task
        task_model = TaskSpecificModel(model, task_index=i)
        task_model.to(device)
        task_model.eval()

        # SHAP DeepExplainer
        explainer = shap.DeepExplainer(
            task_model, 
            torch.tensor(X_sample_scaled, dtype=torch.float32).to(device)
        )

        shap_values = explainer.shap_values(
            torch.tensor(X_sample_scaled, dtype=torch.float32).to(device)
        )

        # shap_values might be a list of arrays
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        print(f"SHAP for {task}: {shap_values.shape}, Data: {X_sample_scaled.shape}")
        if shap_values.shape[1] != X_sample_scaled.shape[1]:
            raise AssertionError(
                "Mismatch between shap_values shape and input data shape."
            )

        shap_values_dict[task] = shap_values

        # Show summary plot
        shap.summary_plot(
            shap_values, 
            features=X_sample_scaled, 
            feature_names=feature_names,
            show=True,
            title=f"SHAP Summary for {task} ({method}-{model_type})",
            sort=False
        )

    return shap_values_dict


##########################################################################
# 7.9 Error Analysis Function
##########################################################################
def error_analysis(predictions, targets, threshold=2.0, 
                   method='evaluation', model_type='shared'):
    """
    Identifies and analyzes cases where absolute % error > threshold.
    Also highlights them on a scatter plot for each task.
    """
    num_tasks = targets.shape[1]
    task_names = ['Energy', 'Cost', 'Emission', 'Comfort']

    for i in range(num_tasks):
        task = task_names[i]
        errors = 100 * np.abs(
            (predictions[:, i] - targets[:, i]) / (np.abs(targets[:, i]) + 1e-6)
        )
        high_error_indices = np.where(errors > threshold)[0]
        print(f"\n{task} Task: {len(high_error_indices)} cases with > {threshold}% error.")

        if len(high_error_indices) > 0:
            sample_indices = high_error_indices[:5]  # show first 5
            print(f"Sample high-error cases for {task} Task:")
            for idx in sample_indices:
                print(
                    f"  Actual: {targets[idx, i]:.2f}, "
                    f"Predicted: {predictions[idx, i]:.2f}, "
                    f"Error: {errors[idx]:.2f}%"
                )

            # Visualization
            plt.figure(figsize=(8, 5))
            plt.scatter(targets[:, i], predictions[:, i], alpha=0.5, label='Normal')
            plt.scatter(
                targets[high_error_indices, i],
                predictions[high_error_indices, i],
                color='red',
                alpha=0.7,
                label='High Error'
            )
            min_val = min(targets[:, i].min(), predictions[:, i].min())
            max_val = max(targets[:, i].max(), predictions[:, i].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(
                f'High Errors - {task} Task\n({method.capitalize()} - {model_type.capitalize()})'
            )
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()


##########################################################################
# 7.10 Ablation Studies
##########################################################################
def ablation_study(model_class, train_loader, val_loader, test_loader, 
                   scalers_Y, feature_indices_to_remove, 
                   input_size=None,  # NOTE: we must pass input_size explicitly
                   num_epochs=50, learning_rate=1e-4, 
                   weights=None, method='weighted_sum', model_type='shared'):
    """
    Performs ablation study by removing specified features and re-training/evaluating.
    :param feature_indices_to_remove: List of feature indices to remove from input.
    :param input_size: The original input feature dimension, needed to compute new input size.
    """
    from .training_functions import (
        train_weighted_sum,
        train_mgda,
        train_uncertainty,
        train_cagrad
    )

    # Helper
    def modify_loader(loader, feature_indices):
        new_inputs = []
        new_targets = []
        for inputs, targets in loader:
            # remove columns in feature_indices
            keep_mask = np.ones(inputs.shape[1], dtype=bool)
            keep_mask[feature_indices] = False
            inputs_mod = inputs[:, keep_mask]
            new_inputs.append(inputs_mod)
            new_targets.append(targets)
        new_inputs = torch.cat(new_inputs, dim=0)
        new_targets = torch.cat(new_targets, dim=0)
        dataset = TensorDataset(new_inputs, new_targets)
        return DataLoader(dataset, batch_size=32, shuffle=True)

    if input_size is None:
        raise ValueError("You must provide the original input_size for ablation_study!")

    # Modify data
    train_loader_mod = modify_loader(train_loader, feature_indices_to_remove)
    val_loader_mod   = modify_loader(val_loader,   feature_indices_to_remove)
    test_loader_mod  = modify_loader(test_loader,  feature_indices_to_remove)

    # Create new model with reduced input dimension
    input_size_mod = input_size - len(feature_indices_to_remove)
    model = model_class(input_size=input_size_mod, hidden_size=256)  # Adjust hidden_size if needed

    # Train
    if method == 'weighted_sum':
        history = train_weighted_sum(
            model, train_loader_mod, val_loader_mod,
            num_epochs, learning_rate, weights,
            method=method, model_type=model_type
        )
    elif method == 'mgda':
        history = train_mgda(
            model, train_loader_mod, val_loader_mod,
            num_epochs, learning_rate,
            method=method, model_type=model_type
        )
    elif method == 'uncertainty':
        history = train_uncertainty(
            model, train_loader_mod, val_loader_mod,
            num_epochs, learning_rate,
            method=method, model_type=model_type
        )
    elif method == 'cagrad':
        history = train_cagrad(
            model, train_loader_mod, val_loader_mod,
            num_epochs, learning_rate,
            method=method, model_type=model_type
        )
    else:
        raise ValueError("Invalid method")

    # Evaluate
    metrics_df, targets_real, predictions_real = evaluate_model_inverse_transform(
        model, test_loader_mod, scalers_Y, 
        method=method, model_type=model_type
    )

    print(f"\nAblation Study - Removed Feature Indices: {feature_indices_to_remove}")
    print(metrics_df)
    return metrics_df, history


##########################################################################
# 7.11 Robustness Testing Function
##########################################################################
def robustness_testing(model, data_loader, scalers_Y, 
                       noise_level=0.1, method='evaluation', model_type='shared'):
    """
    Tests model robustness by adding random Gaussian noise to the inputs.
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            noise = torch.randn_like(inputs) * noise_level
            noisy_inputs = inputs + noise
            outputs = model(noisy_inputs)
            preds = torch.cat(outputs, dim=1)
            all_predictions.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    # Inverse transform
    all_targets_real_list = []
    all_predictions_real_list = []
    for i in range(len(scalers_Y)):
        y_true = all_targets[:, i].reshape(-1, 1)
        y_pred = all_predictions[:, i].reshape(-1, 1)
        scaler = scalers_Y[i]
        y_true_real = scaler.inverse_transform(y_true)
        y_pred_real = scaler.inverse_transform(y_pred)
        all_targets_real_list.append(y_true_real)
        all_predictions_real_list.append(y_pred_real)

    all_targets_real = np.hstack(all_targets_real_list)
    all_predictions_real = np.hstack(all_predictions_real_list)

    # Metrics
    task_names = ['Energy', 'Cost', 'Emission', 'Comfort']
    metrics = { 'Task': task_names, 'MAE': [], 'RMSE': [], 'R2': [] }

    for i in range(len(scalers_Y)):
        mse = mean_squared_error(all_targets_real[:, i], all_predictions_real[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets_real[:, i], all_predictions_real[:, i])
        r2 = r2_score(all_targets_real[:, i], all_predictions_real[:, i])

        metrics['MAE'].append(mae)
        metrics['RMSE'].append(rmse)
        metrics['R2'].append(r2)

    metrics_df = pd.DataFrame(metrics).set_index('Task')
    print(f"\nRobustness Testing with Noise Level {noise_level}:")
    print(metrics_df)

    # Optional: Plot
    for i, task in enumerate(task_names):
        plt.figure(figsize=(6, 6))
        plt.scatter(all_targets_real[:, i], all_predictions_real[:, i], alpha=0.5)
        min_val = min(all_targets_real[:, i].min(), all_predictions_real[:, i].min())
        max_val = max(all_targets_real[:, i].max(), all_predictions_real[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(
            f'Noise = {noise_level}, {task} Task\n({method.capitalize()} - {model_type.capitalize()})'
        )
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return metrics_df


##########################################################################
# 7.12 Real-World Validation
##########################################################################
def real_world_validation(model, data_loader, scalers_Y, 
                          method='evaluation', model_type='shared'):
    """
    Validate the model on test data. 
    Inverse transforms predictions, prints metrics, plots actual vs. predicted.
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.cat(outputs, dim=1)
            all_predictions.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    # Inverse transform
    all_targets_real_list = []
    all_predictions_real_list = []
    for i in range(len(scalers_Y)):
        y_true = all_targets[:, i].reshape(-1, 1)
        y_pred = all_predictions[:, i].reshape(-1, 1)
        scaler = scalers_Y[i]
        y_true_real = scaler.inverse_transform(y_true)
        y_pred_real = scaler.inverse_transform(y_pred)
        all_targets_real_list.append(y_true_real)
        all_predictions_real_list.append(y_pred_real)

    all_targets_real = np.hstack(all_targets_real_list)
    all_predictions_real = np.hstack(all_predictions_real_list)

    task_names = ['Energy', 'Cost', 'Emission', 'Comfort']
    metrics = { 'Task': task_names, 'MAE': [], 'RMSE': [], 'R2': [] }

    for i in range(len(scalers_Y)):
        mae = mean_absolute_error(all_targets_real[:, i], all_predictions_real[:, i])
        mse = mean_squared_error(all_targets_real[:, i], all_predictions_real[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(all_targets_real[:, i], all_predictions_real[:, i])

        metrics['MAE'].append(mae)
        metrics['RMSE'].append(rmse)
        metrics['R2'].append(r2)

    metrics_df = pd.DataFrame(metrics).set_index('Task')
    print(f"\nReal-World Validation Metrics (Test Data) - {method.capitalize()} - {model_type.capitalize()}:")
    print(metrics_df)

    # Quick plot
    plot_actual_vs_predicted(all_targets_real, all_predictions_real, method, model_type)

    return metrics_df, all_targets_real, all_predictions_real
