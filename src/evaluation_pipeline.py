# src/evaluation_pipeline.py

import os
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# If you keep them separate:
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
from .evaluation_functions import (
    evaluate_model_inverse_transform,
    evaluate_model_percentage_error,
    rank_models,
    visualize_ranked_models,
    plot_actual_vs_predicted
)


def evaluate_all_models(
    method_list,
    model_type_list,
    input_size,
    hidden_size,
    test_loader,
    input_features,
    task_names,
    get_model_path_fn=None,
    get_scaler_path_fn=None
):
    """
    Orchestrates the Step 9 evaluation process. Loops over (method, model_type) combos,
    loads trained models + scalers, performs evaluations & plots.

    :param method_list: list of methods (e.g. ['weighted_sum', 'mgda', 'uncertainty']).
    :param model_type_list: list of model types (e.g. ['shared','separate',...]).
    :param input_size: number of features (e.g. 5).
    :param hidden_size: integer (e.g. 256).
    :param test_loader: DataLoader for test data.
    :param input_features: list of input feature names.
    :param task_names: list of task names (e.g. ['Energy','Cost','Emission','Comfort']).
    :param get_model_path_fn: optional override for get_model_path function.
    :param get_scaler_path_fn: optional override for get_scaler_path function.

    :return: 
        evaluation_dict, robustness_metrics_dict, real_world_validation_dict, 
        performance_df (DataFrame of performance metrics), 
        ranked_models (DataFrame of ranked models).
    """

    if get_model_path_fn is None:
        get_model_path_fn = get_model_path
    if get_scaler_path_fn is None:
        get_scaler_path_fn = get_scaler_path

    # We'll store final results in dictionaries
    evaluation_dict = {}
    robustness_metrics_dict = {}
    real_world_validation_dict = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # For easy references
    from .evaluation_functions import (
        error_analysis,
        robustness_testing,
        real_world_validation
    )

    # We also define inline "consolidated" versions of error analysis and robustness 
    # if you want them in a single figure approach:
    def error_analysis_consolidated(predictions, targets, threshold=10.0, method='evaluation', model_type='shared'):
        """
        Identifies & plots high-error cases in a single 2x2 figure for four tasks.
        """
        num_tasks = targets.shape[1]
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = axes.flatten()

        for i in range(num_tasks):
            task = task_names[i]
            errors = 100 * np.abs((predictions[:, i] - targets[:, i]) / (np.abs(targets[:, i]) + 1e-6))
            high_error_indices = np.where(errors > threshold)[0]
            print(f"\n{task} Task: {len(high_error_indices)} cases with > {threshold}% error.")
            sample_indices = high_error_indices[:5]
            print(f"Sample high-error cases for {task} Task:")
            for idx in sample_indices:
                print(f"Actual: {targets[idx, i]:.2f}, Pred: {predictions[idx, i]:.2f}, Error: {errors[idx]:.2f}%")

            axes[i].scatter(targets[:, i], predictions[:, i], alpha=0.5, label='Normal')
            axes[i].scatter(
                targets[high_error_indices, i],
                predictions[high_error_indices, i],
                color='red', alpha=0.7, label='High Error'
            )
            min_val = min(targets[:, i].min(), predictions[:, i].min())
            max_val = max(targets[:, i].max(), predictions[:, i].max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal')
            axes[i].set_xlabel('Actual Values')
            axes[i].set_ylabel('Predicted Values')
            axes[i].set_title(f'High Errors - {task} Task\n({method.capitalize()} - {model_type.capitalize()})')
            axes[i].legend()
            axes[i].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

    def robustness_testing_consolidated(model, data_loader, scalers_Y, noise_level=0.1, method='evaluation', model_type='shared'):
        """
        Tests model robustness with noise, plots 4 tasks in single 2x2 figure.
        """
        model.eval()
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

        # Basic metrics
        task_names_local = task_names  # pass reference
        metrics = { 'Task': task_names_local, 'MAE': [], 'RMSE': [], 'R2': [] }
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

        # Plot in 2x2
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = axes.flatten()
        for i, task in enumerate(task_names_local):
            axes[i].scatter(all_targets_real[:, i], all_predictions_real[:, i], alpha=0.5)
            min_val = min(all_targets_real[:, i].min(), all_predictions_real[:, i].min())
            max_val = max(all_targets_real[:, i].max(), all_predictions_real[:, i].max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')
            axes[i].set_xlabel('Actual')
            axes[i].set_ylabel('Predicted')
            axes[i].set_title(f'Noise={noise_level}, {task}\n({method.capitalize()}-{model_type.capitalize()})')
            axes[i].grid(True)
        plt.tight_layout()
        plt.show()

        return metrics_df

    def real_world_validation_consolidated(model, data_loader, scalers_Y, method='evaluation', model_type='shared'):
        """
        Validate on test data, 2x2 figure for the 4 tasks.
        """
        model.eval()
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
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

        # Basic metrics
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
        print(f"\nReal-World Validation (Test Data) - {method.capitalize()} - {model_type.capitalize()}:")
        print(metrics_df)

        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = axes.flatten()
        for i, task in enumerate(task_names):
            axes[i].scatter(all_targets_real[:, i], all_predictions_real[:, i], alpha=0.5)
            min_val = min(all_targets_real[:, i].min(), all_predictions_real[:, i].min())
            max_val = max(all_targets_real[:, i].max(), all_predictions_real[:, i].max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')
            axes[i].set_xlabel('Actual')
            axes[i].set_ylabel('Predicted')
            axes[i].set_title(f'Actual vs Predicted - {task}\n({method.capitalize()} - {model_type.capitalize()})')
            axes[i].grid(True)
        plt.tight_layout()
        plt.show()

        return metrics_df, all_targets_real, all_predictions_real

    # Lists or placeholders to keep references to final data
    evaluation_dict = {}
    robustness_metrics_dict = {}
    real_world_validation_dict = {}

    for method in method_list:
        for mt in model_type_list:
            print(f"\nEvaluating {mt.capitalize()} Model with {method.capitalize()} Method:")

            # 1) Build the correct model architecture
            #    (Similar logic to your training pipeline)
            if method == 'uncertainty':
                if mt == 'shared':
                    model = SharedMTLModelWithUncertainty(input_size, hidden_size, num_tasks=4)
                elif mt == 'separate':
                    model = SeparateMTLModelWithUncertainty(input_size, hidden_size, num_tasks=4)
                elif mt == 'Ref_Based_Isa':
                    model = Ref_Based(input_size, hidden_size1=128, hidden_size2=64, dropout_rate=0.5, num_tasks=4)
                elif mt == 'Data_Based_Isa':
                    model = Data_Based(
                        input_size=input_size, hidden_size1=128, hidden_size2=64,
                        shared_energy_emission_size=32, shared_comfort_size=32,
                        dropout_rate=0.5, num_tasks=4
                    )
                elif mt == 'More_Shared_Layer':
                    model = More_Shared(
                        input_size, hidden_size1=256, hidden_size2=128, hidden_size3=64,
                        dropout_rate=0.5, num_tasks=4
                    )
                elif mt == 'Few_Shared_Layers':
                    model = Few_Shared(
                        input_size, hidden_size_shared=128, dropout_rate=0.5, num_tasks=4
                    )
                elif mt == 'Deep_Balanced_Layer':
                    model = Deep_Balanced(
                        input_size, hidden_size1=200, hidden_size2=100, hidden_size3=50,
                        dropout_rate=0.5, num_tasks=4
                    )
                else:
                    print(f"Invalid model type {mt}. Skipping.")
                    continue
            elif method == 'cagrad':
                if mt == 'shared':
                    model = SharedMTLModel(input_size, hidden_size)
                elif mt == 'separate':
                    model = SeparateMTLModel(input_size, hidden_size)
                elif mt == 'Ref_Based_Isa':
                    model = Ref_Based(input_size, hidden_size1=128, hidden_size2=64, dropout_rate=0.5, num_tasks=4)
                elif mt == 'Data_Based_Isa':
                    model = Data_Based(
                        input_size=input_size, hidden_size1=128, hidden_size2=64,
                        shared_energy_emission_size=32, shared_comfort_size=32,
                        dropout_rate=0.5, num_tasks=4
                    )
                elif mt == 'More_Shared_Layer':
                    model = More_Shared(
                        input_size, hidden_size1=256, hidden_size2=128, hidden_size3=64,
                        dropout_rate=0.5, num_tasks=4
                    )
                elif mt == 'Few_Shared_Layers':
                    model = Few_Shared(
                        input_size, hidden_size_shared=128, dropout_rate=0.5, num_tasks=4
                    )
                elif mt == 'Deep_Balanced_Layer':
                    model = Deep_Balanced(
                        input_size, hidden_size1=200, hidden_size2=100, hidden_size3=50,
                        dropout_rate=0.5, num_tasks=4
                    )
                else:
                    print(f"Invalid model type {mt}. Skipping.")
                    continue
            else:
                # weighted_sum, mgda, or default
                if mt == 'shared':
                    model = SharedMTLModel(input_size, hidden_size)
                elif mt == 'separate':
                    model = SeparateMTLModel(input_size, hidden_size)
                elif mt == 'Ref_Based_Isa':
                    model = Ref_Based(input_size, hidden_size1=128, hidden_size2=64, dropout_rate=0.5, num_tasks=4)
                elif mt == 'Data_Based_Isa':
                    model = Data_Based(
                        input_size=input_size, hidden_size1=128, hidden_size2=64,
                        shared_energy_emission_size=32, shared_comfort_size=32,
                        dropout_rate=0.5, num_tasks=4
                    )
                elif mt == 'More_Shared_Layer':
                    model = More_Shared(
                        input_size, hidden_size1=256, hidden_size2=128, hidden_size3=64,
                        dropout_rate=0.5, num_tasks=4
                    )
                elif mt == 'Few_Shared_Layers':
                    model = Few_Shared(
                        input_size, hidden_size_shared=128, dropout_rate=0.5, num_tasks=4
                    )
                elif mt == 'Deep_Balanced_Layer':
                    model = Deep_Balanced(
                        input_size, hidden_size1=200, hidden_size2=100, hidden_size3=50,
                        dropout_rate=0.5, num_tasks=4
                    )
                else:
                    print(f"Invalid model type {mt}. Skipping.")
                    continue

            # 2) Load model checkpoint
            model_path = get_model_path_fn(method, mt)
            if not os.path.exists(model_path):
                print(f"Model file {model_path} not found. Skipping.")
                continue
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            
            # 3) Load X-scaler
            scaler_X_path = get_scaler_path_fn(method, mt, scaler_type='X')
            if not os.path.exists(scaler_X_path):
                print(f"Scaler file {scaler_X_path} not found. Skipping.")
                continue
            scaler_X = joblib.load(scaler_X_path)

            # 4) Load Y-scalers
            scalers_Y = []
            for i in range(len(task_names)):
                scaler_Y_path = get_scaler_path_fn(method, mt, scaler_type=f'Y_{i}')
                if not os.path.exists(scaler_Y_path):
                    print(f"Scaler file {scaler_Y_path} not found for task {task_names[i]}. Skipping.")
                    break
                y_scaler = joblib.load(scaler_Y_path)
                scalers_Y.append(y_scaler)
            else:
                # if we didn't break, we proceed
                pass

            if len(scalers_Y) != len(task_names):
                print(f"Incomplete scalers for {mt}_{method}. Skipping.")
                continue

            # 5) Evaluate and store results
            # 9.2.1 Inverse Transform & Metric Calc
            print("\n[Evaluation] Inverse Transform + Metric Calculation")
            metrics_df, targets_real, predictions_real = evaluate_model_inverse_transform(
                model, test_loader, scalers_Y, method=method, model_type=mt
            )
            key = f'{mt}_{method}'
            evaluation_dict[key] = metrics_df

            # 9.2.2 Error Analysis
            print("\nPerforming Error Analysis (Consolidated)...")
            error_analysis_consolidated(predictions_real, targets_real, threshold=10.0, method=method, model_type=mt)

            # 9.2.3 Robustness Testing
            print("\nPerforming Robustness Testing (Consolidated)...")
            robust_df = robustness_testing_consolidated(
                model=model,
                data_loader=test_loader,
                scalers_Y=scalers_Y,
                noise_level=0.1,
                method=method,
                model_type=mt
            )
            robustness_metrics_dict[key] = robust_df

            # 9.2.4 Real-World Validation
            print("\nPerforming Real-World Validation on Test Data (Consolidated)...")
            rwv_df, _, _ = real_world_validation_consolidated(
                model=model,
                data_loader=test_loader,
                scalers_Y=scalers_Y,
                method=method,
                model_type=mt
            )
            real_world_validation_dict[key] = rwv_df

    # 9.2.5 Model Ranking
    print("\nRanking All Models Based on Performance Metrics...")
    ranked_models_df = rank_models(evaluation_dict, weights={'MAE': 0.30, 'RMSE': 0.30, 'R2': 0.40})
    print("\nRanked Models:")
    print(ranked_models_df)

    # 9.2.6 Visualize Ranked Models
    print("\nVisualizing Ranked Models...")
    visualize_ranked_models(ranked_models_df)

    # 9.3 Performance Metrics Table
    print("\nGenerating Performance Metrics Table...")
    performance_data = []

    for model_key, mdf in evaluation_dict.items():
        if mdf is None or mdf.empty:
            continue
        row_dict = {'Model': model_key}
        for task in task_names:
            row_dict[f'{task}_MAE'] = mdf.loc[task, 'MAE']
            row_dict[f'{task}_MSE'] = mdf.loc[task, 'MSE']
            row_dict[f'{task}_RMSE'] = mdf.loc[task, 'RMSE']
            row_dict[f'{task}_R2'] = mdf.loc[task, 'R2']
        # Averages
        row_dict['Average_MAE'] = mdf['MAE'].mean()
        row_dict['Average_MSE'] = mdf['MSE'].mean()
        row_dict['Average_RMSE'] = mdf['RMSE'].mean()
        row_dict['Average_R2'] = mdf['R2'].mean()
        performance_data.append(row_dict)

    performance_df = pd.DataFrame(performance_data).set_index('Model', drop=True)
    print("\n--- Raw Performance Metrics ---")
    print(performance_df)

    # Normalize columns (MAE, MSE, RMSE)
    metrics_to_normalize = ['MAE','MSE','RMSE']
    for metric in metrics_to_normalize:
        for task in task_names:
            col = f"{task}_{metric}"
            norm_col = f"{task}_{metric}_Norm"
            min_val = performance_df[col].min()
            max_val = performance_df[col].max()
            if max_val - min_val == 0:
                performance_df[norm_col] = 1.0
            else:
                performance_df[norm_col] = (performance_df[col] - min_val) / (max_val - min_val)

    # Also average columns
    for metric in metrics_to_normalize:
        avg_col = f"Average_{metric}"
        norm_avg_col = f"Average_{metric}_Norm"
        min_val = performance_df[avg_col].min()
        max_val = performance_df[avg_col].max()
        if max_val - min_val == 0:
            performance_df[norm_avg_col] = 1.0
        else:
            performance_df[norm_avg_col] = (performance_df[avg_col] - min_val) / (max_val - min_val)

    # Reorder columns
    raw_cols = []
    norm_cols = []
    for metric in metrics_to_normalize:
        for task in task_names:
            raw_cols.append(f"{task}_{metric}")
            norm_cols.append(f"{task}_{metric}_Norm")
        raw_cols.append(f"Average_{metric}")
        norm_cols.append(f"Average_{metric}_Norm")

    performance_df = performance_df[raw_cols + norm_cols]

    print("\n--- Comprehensive Performance Metrics with Normalized Values ---")
    print(performance_df)

    # Save if you like
    performance_df.to_csv('performance_metrics_table.csv')
    print("\nPerformance metrics table saved to 'performance_metrics_table.csv'")

    # You can also re-rank if you want
    # e.g. weights = {'MAE':0.3, 'RMSE':0.3, 'R2':0.4}
    # final_ranked = rank_models(evaluation_dict, weights=weights)
    # print("Final Ranked Models:\n", final_ranked)

    return (evaluation_dict, 
            robustness_metrics_dict, 
            real_world_validation_dict, 
            performance_df, 
            ranked_models_df)
