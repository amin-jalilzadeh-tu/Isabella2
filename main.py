#!/usr/bin/env python
"""
main.py
Orchestrates the entire pipeline from data preprocessing through training,
evaluation, inference, optimization, MCDM, and post-processing.

You can run it directly:
  python main.py
"""

import sys
import argparse

# Example imports from your 'src/' modules
from src.data_preprocessing import load_and_preprocess_data
from src.training_pipeline import train_models_if_needed
from src.evaluation_pipeline import evaluate_all_models
from src.inference_pipeline import perform_inference
from src.optimization_pipeline import (
    user_driven_moo,
    constraint_based_moo
)
from src.mcdm_pipeline import perform_mcdm
from src.postprocessing_pipeline import (
    example_postprocessing_11_1,
    example_postprocessing_11_2
)

def parse_args():
    """
    Parse command-line arguments (optional).
    """
    parser = argparse.ArgumentParser(description="Orchestrate the entire pipeline.")
    parser.add_argument("--train", action="store_true", help="Whether to run the training pipeline.")
    parser.add_argument("--evaluate", action="store_true", help="Whether to run the evaluation pipeline.")
    parser.add_argument("--inference", action="store_true", help="Whether to run inference on new data.")
    parser.add_argument("--optimize", action="store_true", help="Whether to run the optimization pipeline (Step 11).")
    parser.add_argument("--mcdm", action="store_true", help="Whether to run MCDM (Step 12).")
    parser.add_argument("--postprocess", action="store_true", help="Whether to run post-processing (Step 13).")
    return parser.parse_args()

def main():
    """
    Main entrypoint that ties all steps together.
    """
    args = parse_args()

    print("=== Step 1: Data Preprocessing ===")
    data_dict = load_and_preprocess_data(output_archives=True)
    # data_dict contains df_inputs, df_outputs, train_loader, val_loader, test_loader, etc.
    # We'll unpack what we need below

    # By default, we might do training if requested
    if args.train:
        print("\n=== Step 2: Optional Training Pipeline ===")
        # e.g., do_training=True to actually train, or skip if we already have models
        train_models_if_needed(
            X_data_normalized=None,  # or your actual normalized data if you want
            scalers_Y=data_dict["scalers_Y"],
            scaler_X=data_dict["scaler_X"],
            train_loader=data_dict["train_loader"],
            val_loader=data_dict["val_loader"],
            do_training=True,  # set based on args
            num_epochs=50,  # or read from config
            learning_rate=1e-4,
            weights=[1.0, 1.0, 1.0, 1.0]
        )
    else:
        print("Skipping training (using existing models).")

    # Evaluate
    if args.evaluate:
        print("\n=== Step 3: Evaluation Pipeline ===")
        # Typically you provide the same method_list, model_type_list, etc.:
        method_list = ['weighted_sum', 'mgda', 'uncertainty']
        model_type_list = ['shared','separate','Ref_Based_Isa','Data_Based_Isa','More_Shared_Layer','Few_Shared_Layers','Deep_Balanced_Layer']
        # Evaluate all
        (evaluation_dict,
         robustness_dict,
         real_world_dict,
         performance_df,
         ranked_models_df) = evaluate_all_models(
            method_list=method_list,
            model_type_list=model_type_list,
            input_size=5,      # or data_dict["X_data_normalized"].shape[1]
            hidden_size=256,
            test_loader=data_dict["test_loader"],
            input_features=["time_horizon","windows_U_Factor","groundfloor_thermal_resistance","ext_walls_thermal_resistance","roof_thermal_resistance"],
            task_names=['Energy','Cost','Emission','Comfort']
         )
    else:
        print("Skipping evaluation pipeline.")

    # Inference
    if args.inference:
        print("\n=== Step 4: Inference on New Data ===")
        # Example method & model type
        method = 'uncertainty'
        model_type = 'Data_Based_Isa'
        new_data_input_features = ["time_horizon","windows_U_Factor","groundfloor_thermal_resistance","ext_walls_thermal_resistance","roof_thermal_resistance"]
        # Suppose you have a df_new_data with columns above
        df_new_data = data_dict["df_inputs"].copy()  # or your actual new data
        # Perform inference
        predictions = perform_inference(
            method=method,
            model_type=model_type,
            input_features=new_data_input_features,
            num_targets=4,
            input_size=5,   # adjust
            hidden_size=256,
            df_inputs=df_new_data
        )
        print("Inference predictions:", predictions)
    else:
        print("Skipping inference.")

    # Optimization (Approach 11.1 & 11.2)
    if args.optimize:
        print("\n=== Step 5: Optimization Pipeline ===")

        # 11.1: Suppose you already have 'predictions' from inference + 'df_inputs'
        # user-driven (Pareto approach)
        if "df_inputs" in data_dict and "predictions" in locals():
            df_pareto_11_1 = user_driven_moo(predictions, data_dict["df_inputs"])
            print("User-driven Pareto DF (11.1):")
            print(df_pareto_11_1.head())
        else:
            print("Skipping Approach 11.1 because predictions or df_inputs not available.")

        # 11.2: Constraint-based
        df_pareto_11_2 = constraint_based_moo(
            method='uncertainty',
            model_type='Data_Based_Isa',
            input_size=5,
            hidden_size=256,
            num_targets=4,
            n_generations=100,
            pop_size=50
        )
        print("Constraint-based Pareto DF (11.2):")
        print(df_pareto_11_2.head())

    # MCDM
    if args.mcdm:
        print("\n=== Step 6: Multi-Criteria Decision Making (MCDM) ===")
        # e.g., if we have df_pareto_11_1 & df_pareto_11_2 from the optimization step
        # For demonstration only if they exist
        # ...
        # from src.mcdm_pipeline import perform_mcdm
        # sorted_11_1, best_asf_11_1, best_pseudo_11_1, tradeoff_11_1 = perform_mcdm(df_pareto_11_1, approach='11_1')
        # ...
        print("Skipping actual code here. Just demonstrating location.")
    else:
        print("Skipping MCDM pipeline.")

    # Post-processing & Visualization
    if args.postprocess:
        print("\n=== Step 7: Post-Processing & Visualization ===")
        # Suppose we have 'df_pareto_11_1_with_score' etc. from MCDM
        # from src.postprocessing_pipeline import example_postprocessing_11_1
        # ...
        print("Skipping code. Just demonstrating location.")
    else:
        print("Skipping post-processing.")

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
