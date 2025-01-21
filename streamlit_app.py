#!/usr/bin/env python

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Example imports from your 'src/' modules:
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
    plot_3d_scatter,
    plot_clusters,
    plot_parallel_coordinates,
    perform_sensitivity_analysis,
    plot_correlation_matrix,
    plot_pareto_front_with_mcdm,
    cluster_scenarios,
    plot_density_based_pareto
)

###############################################################################
# HELPER: Save/Load session state to JSON, ensuring DataFrame is serializable
###############################################################################
def save_session_state_to_json(json_path="session_state.json"):
    data_to_save = {}
    # Keys we want to store
    keys_to_store = [
        "df_pareto_11_1",
        "df_pareto_11_2",
        "mcdm_11_1",
        "mcdm_11_2",
        "train_done"
    ]
    for k in keys_to_store:
        if k in st.session_state and st.session_state[k] is not None:
            val = st.session_state[k]
            if isinstance(val, pd.DataFrame):
                data_to_save[k] = {
                    "_type": "DataFrame",
                    "_value": val.to_dict(orient="records")
                }
            elif isinstance(val, tuple):
                tuple_list = []
                for item in val:
                    if isinstance(item, pd.DataFrame):
                        tuple_list.append({
                            "_type": "DataFrame",
                            "_value": item.to_dict(orient="records")
                        })
                    else:
                        tuple_list.append(item)
                data_to_save[k] = {
                    "_type": "tuple",
                    "_value": tuple_list
                }
            else:
                data_to_save[k] = val
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    st.success(f"Session saved to {json_path}")

def load_session_state_from_json(json_path="session_state.json"):
    if not os.path.exists(json_path):
        st.warning(f"No session file found at {json_path}")
        return
    with open(json_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    for k,v in loaded_data.items():
        if isinstance(v, dict) and "_type" in v:
            if v["_type"] == "DataFrame":
                st.session_state[k] = pd.DataFrame(v["_value"])
            elif v["_type"] == "tuple":
                restored_list = []
                for item in v["_value"]:
                    if isinstance(item, dict) and item.get("_type") == "DataFrame":
                        restored_list.append(pd.DataFrame(item["_value"]))
                    else:
                        restored_list.append(item)
                st.session_state[k] = tuple(restored_list)
        else:
            st.session_state[k] = v
    st.success(f"Session loaded from {json_path}")

###############################################################################
def main():
    st.set_page_config(page_title="Ultra MTL UI", layout="wide")
    st.title(" Building Retrofit Multi-Task Learning Portal")

    # ADDING NEW PAGE: Introduction
    pages = [
        "Introduction",
        "Data Preprocessing",
        "Training",
        "Evaluation",
        "Inference",
        "Optimization",
        "MCDM",
        "Advanced Post-Processing",
        "Results & Comparison"
    ]
    page = st.sidebar.radio("Navigation", pages)

    if "data_dict" not in st.session_state:
        st.session_state["data_dict"] = None
    if "train_done" not in st.session_state:
        st.session_state["train_done"] = False
    if "df_pareto_11_1" not in st.session_state:
        st.session_state["df_pareto_11_1"] = None
    if "df_pareto_11_2" not in st.session_state:
        st.session_state["df_pareto_11_2"] = None
    if "mcdm_11_1" not in st.session_state:
        st.session_state["mcdm_11_1"] = None
    if "mcdm_11_2" not in st.session_state:
        st.session_state["mcdm_11_2"] = None

    st.sidebar.write("Session Management")
    col_save, col_load = st.sidebar.columns(2)
    if col_save.button("Save Session"):
        save_session_state_to_json("session_state.json")
    if col_load.button("Load Session"):
        load_session_state_from_json("session_state.json")

    ########################################################################
    # PAGE: Introduction
    ########################################################################
    if page == "Introduction":
        st.header("Welcome to the Multi-Task Learning Interface")
        st.write("This tool allows to select optimal retrofit solutions considering the effect of heat waves in the Netherlands. Given the preference of the user, it select the best-performing intervention considering trade offs between energy consumption, comfort, cost and environmental impact of the intervention. It is the result of the Graduation Project of Isabella Lucchese, a student at Delft University of Technology founded by ARUP as part of MultiCare project.")
        st.write("Key contributors to this tool are:")
        st.write("Isabella Lucchese - Author")
        st.write("Amin Jalilzadeh - Supervisor")
        st.write("Azarakhsh Rafiee - First mentor")
        st.write("Willem van der Spoel - Second mentor")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("arup.png", caption="Contributor", width= 100)
        with col2:
            st.image("tudelft.png", caption="Contributor", width=100)
        with col3:
            st.image("multicare.png", caption="Contributor", width=130)
        st.write(
            """
        # **Multitask Learning Pipeline for Building Performance Analysis**

        This code implements a comprehensive workflow for training, evaluating, and optimizing a **Multitask Learning (MTL)** model to predict critical building performance indicators. By combining different modules—from data preprocessing and model architecture definitions to multi-objective optimization and post-processing—this pipeline provides a seamless end-to-end solution for retrofit decision-making.

        ---

        ### **Key Highlights**

        - **Data Loading & Preprocessing**  
        - Reads simulation outputs for multiple time horizons (2020, 2050, 2100).  
        - Automatically merges and cleans data (e.g., checking if "Simulation ID" is consistent).  
        - Splits data into inputs (e.g., U-factor, thermal resistance) and outputs (energy, carbon, cost, comfort).  
        - Normalizes both inputs and outputs with `MinMaxScaler` to stabilize training.

        - **Model Architectures**  
        - Offers various neural network designs (e.g., **SharedMTLModel**, **SeparateMTLModel**, **Ref_Based**, **Data_Based**) for flexible experimentation.  
        - Implements uncertainty-based, MGDA, and weighted-sum methods to handle multi-task objectives.  
        - Accommodates unique branches for different targets—Annual Energy Consumption, Total Retrofit Cost, CO₂ Emissions, Comfort Days.

        - **Training & Early Stopping**  
        - Each method (weighted-sum, MGDA, uncertainty, CAGrad) has a dedicated training routine.  
        - **EarlyStopping** halts training when validation loss fails to improve, helping avoid overfitting.

        - **Evaluation & Metrics**  
        - Includes inverse-transform evaluations to revert scaled predictions to their original magnitude.  
        - Computes MAE, RMSE, R², and more.  
        - Offers percentage error histograms and actual vs. predicted plots for transparent model performance insight.

        - **Multi-Objective Optimization**  
        - Explores **Pareto efficiency**: balancing trade-offs among conflicting objectives (energy, cost, carbon, comfort).  
        - Utilizes the **Pymoo** library algorithms (e.g., NSGA2) to identify non-dominated solutions under given constraints (minimum/maximum U-factor, thermal resistance ranges).  
        - Combines user-driven (11.1) and constraint-based (11.2) approaches to yield flexible retrofit strategies.

        - **MCDM (Multi-Criteria Decision Making)**  
        - Applies Weighted Scores, ASF, Pseudo-Weights, and High Trade-off Points to prioritize Pareto-optimal solutions.  
        - Helps stakeholders choose a “best” design based on specific weighting preferences (e.g., more emphasis on reducing cost vs. maximizing comfort).

        - **Post-Processing & Visualization**  
        - Uses clustering (K-Means) and correlation analysis to group similar solutions and reveal design trends.  
        - Generates parallel coordinates, 3D scatter plots, and heatmaps to interpret how input features influence each target.  
        - Provides sensitivity analysis via linear regression coefficients to quantify which parameters matter most.

        ---

        **In essence,** this code enables users to ingest building simulation data, train specialized MTL neural networks, evaluate their predictive fidelity, and then conduct multi-objective optimizations. Decision-makers can thus refine retrofitting strategies, balancing energy efficiency, cost, carbon footprint, and occupant comfort in an integrated, data-driven manner.
        """
        )

    ########################################################################
    # PAGE: Data Preprocessing
    ########################################################################
    elif page == "Data Preprocessing":
        st.header("Step 1: Data Preprocessing")
        if st.button("Run Data Preprocessing"):
            out_arch = st.checkbox("Save CSVs?", True)
            data_dict = load_and_preprocess_data(output_archives=out_arch)
            st.session_state["data_dict"] = data_dict
            st.success("Data Preprocessing Done")
            st.write("df_inputs:", data_dict["df_inputs"].shape)
            st.write("df_outputs:", data_dict["df_outputs"].shape)
        if st.session_state["data_dict"] is not None:
            st.write("Preview df_inputs:")
            st.dataframe(st.session_state["data_dict"]["df_inputs"].head())
            st.write("Preview df_outputs:")
            st.dataframe(st.session_state["data_dict"]["df_outputs"].head())

    ########################################################################
    # PAGE: Training
    ########################################################################
    elif page == "Training":
        st.header("Step 2: Training")
        if st.session_state["data_dict"] is None:
            st.warning("Preprocessing needed first.")
        else:
            do_train = st.checkbox("Actually train now?", True)
            epochs = st.slider("Epochs", 10, 300, 50, 10)
            lr = st.number_input("Learning rate", value=1e-4, format="%.1e")
            st.subheader("Weighted Sum Weights (if Weighted Sum used)")
            c1,c2,c3,c4 = st.columns(4)
            w_e = c1.number_input("Energy Weight", 0.0, 10.0, 1.0, 0.1)
            w_c = c2.number_input("Cost Weight", 0.0, 10.0, 1.0, 0.1)
            w_m = c3.number_input("Emission Weight", 0.0, 10.0, 1.0, 0.1)
            w_f = c4.number_input("Comfort Weight", 0.0, 10.0, 1.0, 0.1)
            if st.button("Run Training"):
                wsum = [w_e, w_c, w_m, w_f]
                train_models_if_needed(
                    X_data_normalized=None,
                    scalers_Y=st.session_state["data_dict"]["scalers_Y"],
                    scaler_X=st.session_state["data_dict"]["scaler_X"],
                    train_loader=st.session_state["data_dict"]["train_loader"],
                    val_loader=st.session_state["data_dict"]["val_loader"],
                    do_training=do_train,
                    num_epochs=epochs,
                    learning_rate=lr,
                    weights=wsum
                )
                st.session_state["train_done"] = True
                st.success("Training completed")

    ########################################################################
    # PAGE: Evaluation
    ########################################################################
    elif page == "Evaluation":
        st.header("Step 3: Evaluation")
        if st.session_state["data_dict"] is None:
            st.warning("No data for evaluation.")
        else:
            st.write("Select methods to evaluate:")
            methods_sel = st.multiselect("Methods", ["weighted_sum","mgda","uncertainty","cagrad"], default=["weighted_sum","mgda"])
            if st.button("Run Evaluation"):
                model_types = ["shared","separate","Ref_Based_Isa","Data_Based_Isa","More_Shared_Layer","Few_Shared_Layers","Deep_Balanced_Layer"]
                (eval_dict, robust_dict, real_dict, perf_df, rank_df) = evaluate_all_models(
                    method_list=methods_sel,
                    model_type_list=model_types,
                    input_size=5,
                    hidden_size=256,
                    test_loader=st.session_state["data_dict"]["test_loader"],
                    input_features=[
                        "time_horizon","windows_U_Factor","groundfloor_thermal_resistance",
                        "ext_walls_thermal_resistance","roof_thermal_resistance"
                    ],
                    task_names=["Energy","Cost","Emission","Comfort"]
                )
                st.write("Performance DF:")
                st.dataframe(perf_df)
         #       st.write("Ranked Models:")
         #       st.dataframe(rank_df)

    ########################################################################
    # PAGE: Inference
    ########################################################################
    elif page == "Inference":
        st.header("Step 4: Inference")
        upfile = st.file_uploader("Upload CSV for inference", type=["csv"])
        if upfile:
            df_user = pd.read_csv(upfile)
            st.dataframe(df_user.head())
        else:
            if st.session_state["data_dict"]:
                df_user = st.session_state["data_dict"]["df_inputs"].copy()
                st.write("Using df_inputs fallback.")
            else:
                df_user = None
        method_sel = st.selectbox("Method", ["weighted_sum","mgda","uncertainty","cagrad"], index=2)
        model_sel = st.selectbox("Model Type", ["shared","separate","Ref_Based_Isa","Data_Based_Isa","More_Shared_Layer","Few_Shared_Layers","Deep_Balanced_Layer"], index=3)
        if st.button("Run Inference"):
            if df_user is None:
                st.warning("No data found.")
            else:
                feats = [
                    "time_horizon","windows_U_Factor","groundfloor_thermal_resistance",
                    "ext_walls_thermal_resistance","roof_thermal_resistance"
                ]
                preds = perform_inference(
                    method_sel,
                    model_sel,
                    feats,
                    4,
                    len(feats),
                    256,
                    df_user
                )
                st.success(f"Inference done on {method_sel}_{model_sel}")
                df_pred = pd.DataFrame(preds, columns=["Energy","Cost","Emission","Comfort"])
                st.dataframe(df_pred.head(20))

    ########################################################################
    # PAGE: Optimization
    ########################################################################
    elif page == "Optimization":
        st.header("Step 5: Optimization")
        st.subheader("Approach 11.1 - Optimization based on dataset")
        if st.button("Run Approach 11.1"):
            if st.session_state["data_dict"] is None:
                st.warning("Need data.")
            else:
                df_in = st.session_state["data_dict"]["df_inputs"]
                feats = [
                    "time_horizon","windows_U_Factor","groundfloor_thermal_resistance",
                    "ext_walls_thermal_resistance","roof_thermal_resistance"
                ]
                p_11_1 = perform_inference(
                    method="uncertainty",
                    model_type="Data_Based_Isa",
                    input_features=feats,
                    num_targets=4,
                    input_size=len(feats),
                    hidden_size=256,
                    df_inputs=df_in
                )
                df_p11_1 = user_driven_moo(p_11_1, df_in)
                st.session_state["df_pareto_11_1"] = df_p11_1
                st.dataframe(df_p11_1.head())

        st.subheader("Approach 11.2 - Optimization based on contraints")
        t_hor = st.slider("Time horizon", 2020, 2100, (2020,2100), 10)
        w_minmax = st.slider("Windows U-Factor", 0.0, 10.0, (0.2,2.0), 0.1)
        gf_minmax = st.slider("Ground Floor R", 0.0, 10.0, (0.5,5.0), 0.1)
        ew_minmax = st.slider("Ext Walls R", 0.0, 10.0, (0.5,5.0), 0.1)
        rf_minmax = st.slider("Roof R", 0.0, 10.0, (0.5,5.0), 0.1)
        if st.button("Run Approach 11.2"):
            df_11_2 = constraint_based_moo(
                method="uncertainty",
                model_type="Data_Based_Isa",
                input_size=5,
                hidden_size=256,
                num_targets=4,
                n_generations=50,
                pop_size=20
            )
            st.session_state["df_pareto_11_2"] = df_11_2
            st.dataframe(df_11_2.head(10))

    ########################################################################
    # PAGE: MCDM
    ########################################################################
    elif page == "MCDM":
        st.header("Step 6: MCDM")
        approach_choice = st.selectbox("Approach", ["11_1","11_2"])
        c1,c2,c3,c4=st.columns(4)
        wA = c1.number_input("Energy W", 0.0,1.0,0.3,0.05)
        wC = c2.number_input("CO2 W", 0.0,1.0,0.3,0.05)
        wT = c3.number_input("Cost W", 0.0,1.0,0.2,0.05)
        wN = c4.number_input("NegComfort W", 0.0,1.0,0.2,0.05)
        if st.button("Perform MCDM"):
            df_key = "df_pareto_"+approach_choice
            if st.session_state[df_key] is None:
                st.warning(f"No Pareto for {approach_choice}.")
            else:
                from pymoo.decomposition.asf import ASF
                from pymoo.mcdm.pseudo_weights import PseudoWeights
                from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
                df_p = st.session_state[df_key].copy()
                F = df_p[[
                    "Annual Energy Consumption",
                    "Total CO2 Emission",
                    "Total Retrofit Cost",
                    "Negative Comfort Days"
                ]].values
                F_min = F.min(axis=0); F_max = F.max(axis=0)
                F_norm = (F - F_min)/(F_max - F_min)
                w_arr = np.array([wA,wC,wT,wN])
                wscore = F_norm @ w_arr
                df_p["Weighted_Score"] = wscore
                df_sorted = df_p.sort_values(by="Weighted_Score").reset_index(drop=True)
                asf = ASF()
                idx_asf = asf.do(F, w_arr).argmin()
                best_asf = df_p.iloc[idx_asf]
                pw = PseudoWeights(w_arr)
                idx_pw,_=pw.do(F, return_pseudo_weights=True)
                best_pw = df_p.iloc[idx_pw]
                ht = HighTradeoffPoints()
                idx_ht = ht.do(F)
                best_ht = df_p.iloc[idx_ht]
                st.dataframe(df_sorted.head(10))
                st.write("Best ASF:")
                st.dataframe(best_asf.to_frame().T)
#                st.write("Best Pseudo-Weights:")
#                st.dataframe(best_pw.to_frame().T)
                st.write("High Trade-off Points:")
                st.dataframe(best_ht.head(5))
                mcdm_key = "mcdm_"+approach_choice
                st.session_state[mcdm_key] = (df_sorted, best_asf, best_pw, best_ht)

    ########################################################################
    # PAGE: Advanced Post-Processing
    ########################################################################
    elif page == "Advanced Post-Processing":
        st.header("Step 7: Advanced Post-Processing")
        approach_choice = st.selectbox("Approach to post-process", ["11_1","11_2"])
        df_key = "df_pareto_"+approach_choice
        if st.session_state[df_key] is None:
            st.warning("No data. Run optimization first.")
        else:
            df_pp = st.session_state[df_key].copy()
            st.dataframe(df_pp.head(10))

            if st.checkbox("K-Means Clustering"):
                n_c = st.slider("Number of clusters", 2, 10, 5)
                if st.button("Apply KMeans"):
                    df_pp = cluster_scenarios(df_pp, n_clusters=n_c)
                    st.dataframe(df_pp.head(10))
                    fig3d = px.scatter_3d(
                        df_pp,
                        x="Annual Energy Consumption",
                        y="Total CO2 Emission",
                        z="Total Retrofit Cost",
                        color="cluster",
                        hover_data=df_pp.columns,
                        title=f"3D scatter (k={n_c})"
                    )
                    st.plotly_chart(fig3d, use_container_width=True)

            if st.checkbox("Custom 3D Scatter"):
                x_col = st.selectbox("X", df_pp.columns, index=5)
                y_col = st.selectbox("Y", df_pp.columns, index=6)
                z_col = st.selectbox("Z", df_pp.columns, index=7)
                color_by = st.selectbox("Color by", df_pp.columns, index=8)
                fig_3d2 = px.scatter_3d(
                    df_pp,
                    x=x_col,
                    y=y_col,
                    z=z_col,
                    color=color_by,
                    hover_data=df_pp.columns
                )
                st.plotly_chart(fig_3d2, use_container_width=True)

            if st.checkbox("Correlation Matrix"):
                fig_corr, ax_corr = plt.subplots(figsize=(10,8))
                cmat = df_pp.corr()
                sns.heatmap(cmat, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
                st.pyplot(fig_corr)

            if st.checkbox("Parallel Coordinates"):
                inc_cols = st.multiselect("Columns", df_pp.columns.tolist(), default=df_pp.columns.tolist())
                if len(inc_cols)>1:
                    scaled = MinMaxScaler().fit_transform(df_pp[inc_cols])
                    df_scaled = pd.DataFrame(scaled, columns=inc_cols)
                    color_dim = st.selectbox("Color dimension", inc_cols, index=0)
                    fig_par = px.parallel_coordinates(
                        df_scaled,
                        dimensions=inc_cols,
                        color=color_dim,
                        color_continuous_scale=px.colors.sequential.Viridis
                    )
                    st.plotly_chart(fig_par, use_container_width=True)

            if st.checkbox("Sensitivity Analysis"):
                target_col = st.selectbox("Target col", df_pp.columns, index=5)
                coefs = perform_sensitivity_analysis(
                    df_pp,
                    input_features=[
                        "Time Horizon","Windows U-Factor","Ground Floor Thermal Resistance",
                        "External Walls Thermal Resistance","Roof Thermal Resistance"
                    ],
                    target=target_col
                )
                st.write("Coefficients:", coefs)

            if st.checkbox("Density-Based Pareto"):
                x_ax = st.selectbox("X axis", df_pp.columns, index=5)
                y_ax = st.selectbox("Y axis", df_pp.columns, index=6)
                if "Weighted_Score" in df_pp.columns:
                    ws = df_pp["Weighted_Score"].values
                else:
                    ws = np.random.rand(len(df_pp))
                plot_density_based_pareto(
                    df_pp,
                    ws,
                    x=x_ax,
                    y=y_ax,
                    title="Density Based Pareto"
                )

    ########################################################################
    # PAGE: Results & Comparison
    ########################################################################
    elif page == "Results & Comparison":
        st.header("Step 8: Results & Comparison")
        cL, cR = st.columns(2)
        with cL:
            st.subheader("Approach 11.1 MCDM")
            if st.session_state["mcdm_11_1"] is not None:
                df_s11, asf11, pseudo11, ht11 = st.session_state["mcdm_11_1"]
                st.write("Best ASF:")
                st.dataframe(asf11.to_frame().T)
 #               st.write("Best Pseudo-Weights:")
 #               st.dataframe(pseudo11.to_frame().T)
 #               st.write("High Trade-off (top5):")
 #               st.dataframe(ht11.head(5))
            else:
                st.info("No MCDM for 11.1")

        with cR:
            st.subheader("Approach 11.2 MCDM")
            if st.session_state["mcdm_11_2"] is not None:
                df_s22, asf22, pseudo22, ht22 = st.session_state["mcdm_11_2"]
                st.write("Best ASF:")
                st.dataframe(asf22.to_frame().T)
 #               st.write("Best Pseudo-Weights:")
 #               st.dataframe(pseudo22.to_frame().T)
 #               st.write("High Trade-off (top5):")
 #               st.dataframe(ht22.head(5))
            else:
                st.info("No MCDM for 11.2")


if __name__ == "__main__":
    main()
