# src/postprocessing_pipeline.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import joblib
from itertools import combinations
from pandas.plotting import parallel_coordinates

# Pymoo classes if needed for some advanced plotting
from pymoo.decomposition.asf import ASF
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints

# Define constants or pass them in via function params
OBJECTIVE_NAMES = [
    'Annual Energy Consumption',
    'Total CO2 Emission',
    'Total Retrofit Cost',
    'Negative Comfort Days'
]

INPUT_FEATURES = [
    'Time Horizon',
    'Windows U-Factor',
    'Ground Floor Thermal Resistance',
    'External Walls Thermal Resistance',
    'Roof Thermal Resistance'
]

def plot_pareto_front_with_mcdm(pareto_df, weighted_scores, best_asf, best_pseudo, tradeoff_solutions):
    """
    Plots multiple objective pairs (2D) from a Pareto front, highlighting
    best solutions (ASF, Pseudo-Weights, High Trade-off).
    """
    from itertools import combinations
    objective_pairs = list(combinations(OBJECTIVE_NAMES, 2))
    num_pairs = len(objective_pairs)

    cols = 3
    rows = (num_pairs + cols - 1) // cols  # enough rows to fit all pairs
    fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 6*rows))
    axes = axes.flatten()

    for idx, (obj1, obj2) in enumerate(objective_pairs):
        ax = axes[idx]
        scatter = ax.scatter(
            pareto_df[obj1],
            pareto_df[obj2],
            c=weighted_scores,
            cmap='viridis',
            s=50,
            alpha=0.9,
            label='Pareto Front'
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Weighted Score')

        # Plot special points
        ax.scatter(best_asf[obj1], best_asf[obj2], color='red', marker='*', s=250, label='Best ASF')
        ax.scatter(best_pseudo[obj1], best_pseudo[obj2], color='blue', marker='D', s=150, label='Best Pseudo-Weights')
        ax.scatter(tradeoff_solutions[obj1], tradeoff_solutions[obj2], color='orange', marker='X', s=150, label='High Trade-off')

        ax.set_xlabel(obj1)
        ax.set_ylabel(obj2)
        ax.set_title(f'Pareto: {obj1} vs {obj2}')
        ax.grid(True)
        ax.legend()

    # Remove any extra subplots
    for idx in range(num_pairs, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


def cluster_scenarios(predictions_df, n_clusters=5, objective_names=OBJECTIVE_NAMES):
    """
    Applies K-Means clustering on the specified objective columns.
    Returns the DataFrame with an added 'cluster' column.
    """
    cluster_data = predictions_df[objective_names]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    predictions_df['cluster'] = kmeans.fit_predict(cluster_data)
    return predictions_df


def plot_clusters(predictions_df, objective_names=OBJECTIVE_NAMES):
    """
    Plots a seaborn pairplot of the objectives, colored by cluster label.
    """
    sns.pairplot(
        predictions_df,
        vars=objective_names,
        hue='cluster',
        palette='Set1',
        diag_kind='kde',
        plot_kws={'alpha': 0.7}
    )
    plt.suptitle('Clustering of Retrofitting Scenarios', y=1.02)
    plt.show()


def perform_sensitivity_analysis(predictions_df, input_features=INPUT_FEATURES, target='Annual Energy Consumption'):
    """
    Fits a linear regression on the specified target to gauge feature sensitivity.
    Prints and returns the coefficients.
    """
    X = predictions_df[input_features]
    y = predictions_df[target]
    
    lr = LinearRegression()
    lr.fit(X, y)
    coeffs = pd.Series(lr.coef_, index=input_features)
    print(f"Sensitivity Analysis for {target}:")
    print(coeffs)
    return coeffs


def plot_correlation_matrix(pareto_df, input_features=INPUT_FEATURES, objective_names=OBJECTIVE_NAMES):
    """
    Plots a correlation matrix heatmap of both the input features and objectives, plus Weighted_Score if present.
    """
    columns_to_check = [col for col in input_features if col in pareto_df.columns] \
                       + [col for col in objective_names if col in pareto_df.columns]
    if 'Weighted_Score' in pareto_df.columns:
        columns_to_check.append('Weighted_Score')

    corr_data = pareto_df[columns_to_check]
    corr_matrix = corr_data.corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_3d_scatter(predictions_df,
                    x='Annual Energy Consumption',
                    y='Total Retrofit Cost',
                    z='Total CO2 Emission',
                    color='Comfort Days',
                    hover_data=None,
                    title='3D Scatter Plot of Retrofitting Scenarios'):
    """
    Creates a 3D scatter plot using Plotly for better interactivity.
    """
    import plotly.express as px

    if hover_data is None:
        # default columns to show on hover
        hover_data = INPUT_FEATURES + ['Weighted_Score'] if 'Weighted_Score' in predictions_df.columns else INPUT_FEATURES

    fig = px.scatter_3d(
        predictions_df,
        x=x,
        y=y,
        z=z,
        color=color,
        size=color,
        hover_data=hover_data,
        title=title,
        color_continuous_scale='Rainbow',
        size_max=18
    )
    fig.update_layout(scene=dict(
        xaxis_title=x,
        yaxis_title=y,
        zaxis_title=z
    ))
    fig.show()


def plot_parallel_coordinates(pareto_df,
                              input_features=INPUT_FEATURES,
                              objective_names=OBJECTIVE_NAMES,
                              cluster_column='cluster',
                              title='Parallel Coordinates Plot'):
    """
    Creates a parallel coordinates plot using Plotly to compare multi-dimensional data.
    """
    import plotly.express as px

    scaler_vis = MinMaxScaler()
    columns_for_scaling = [col for col in (input_features + objective_names) if col in pareto_df.columns]
    if 'Weighted_Score' in pareto_df.columns:
        columns_for_scaling.append('Weighted_Score')

    scaled_data = scaler_vis.fit_transform(pareto_df[columns_for_scaling])
    scaled_df = pd.DataFrame(scaled_data, columns=columns_for_scaling)

    if cluster_column in pareto_df.columns:
        scaled_df['Cluster'] = pareto_df[cluster_column]
    else:
        scaled_df['Cluster'] = 0  # fallback if cluster isn't present

    fig = px.parallel_coordinates(
        scaled_df,
        dimensions=columns_for_scaling,
        color='Weighted_Score' if 'Weighted_Score' in scaled_df.columns else None,
        color_continuous_scale=px.colors.sequential.Viridis,
        color_continuous_midpoint=scaled_df['Weighted_Score'].mean() if 'Weighted_Score' in scaled_df.columns else 0,
        title=title
    )
    fig.show()


def plot_density_based_pareto(pareto_df,
                              weighted_scores,
                              x='Annual Energy Consumption',
                              y='Total CO2 Emission',
                              title='Density-Based Pareto Front'):
    """
    Uses seaborn's kdeplot to create a density-based visualization,
    then overlays the Weighted_Score-colored scatter points.
    """
    plt.figure(figsize=(12, 8))
    sns.kdeplot(
        x=pareto_df[x],
        y=pareto_df[y],
        cmap="Blues",
        shade=True,
        bw_adjust=0.5,
        alpha=0.5
    )
    scatter = plt.scatter(
        pareto_df[x],
        pareto_df[y],
        c=weighted_scores,
        cmap='viridis',
        s=50,
        alpha=0.9,
        edgecolor='k'
    )
    plt.colorbar(scatter, label='Weighted Score')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title, fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


##########################################################################
# Example "main" function showing how to use these with df_pareto_11_1, etc.
##########################################################################
def example_postprocessing_11_1(df_pareto_11_1_with_score,
                                best_asf_11_1,
                                best_pseudo_11_1,
                                tradeoff_11_1):
    """
    Example pipeline for post-processing and visualization for Approach 11.1
    """
    # We expect df_pareto_11_1_with_score to have Weighted_Score, Negative Comfort Days, etc.
    # 1) Convert Negative Comfort Days -> Comfort Days (if needed)
    if 'Negative Comfort Days' in df_pareto_11_1_with_score.columns and 'Comfort Days' not in df_pareto_11_1_with_score.columns:
        df_pareto_11_1_with_score['Comfort Days'] = -df_pareto_11_1_with_score['Negative Comfort Days']

    # 2) Plot Pareto with MCDM highlights
    weighted_scores_11_1 = df_pareto_11_1_with_score['Weighted_Score'].values
    plot_pareto_front_with_mcdm(
        pareto_df=df_pareto_11_1_with_score,
        weighted_scores=weighted_scores_11_1,
        best_asf=best_asf_11_1,
        best_pseudo=best_pseudo_11_1,
        tradeoff_solutions=tradeoff_11_1
    )

    # 3) Perform clustering
    df_pareto_11_1_with_score = cluster_scenarios(df_pareto_11_1_with_score, n_clusters=5)

    # 4) Plot clusters
    plot_clusters(df_pareto_11_1_with_score)

    # 5) Sensitivity analysis
    perform_sensitivity_analysis(df_pareto_11_1_with_score)

    # 6) Correlation matrix
    plot_correlation_matrix(df_pareto_11_1_with_score)

    # 7) 3D scatter
    plot_3d_scatter(df_pareto_11_1_with_score)

    # 8) Parallel coordinates
    plot_parallel_coordinates(df_pareto_11_1_with_score)

    # 9) [Optional] Density-based Pareto
    # plot_density_based_pareto(df_pareto_11_1_with_score,
    #                           weighted_scores_11_1,
    #                           title='Density-Based Pareto - Approach 11.1')


def example_postprocessing_11_2(df_pareto_11_2_with_score,
                                best_asf_11_2,
                                best_pseudo_11_2,
                                tradeoff_11_2):
    """
    Example pipeline for post-processing and visualization for Approach 11.2
    """
    # If needed, convert Negative Comfort Days -> Comfort Days
    if 'Negative Comfort Days' in df_pareto_11_2_with_score.columns and 'Comfort Days' not in df_pareto_11_2_with_score.columns:
        df_pareto_11_2_with_score['Comfort Days'] = -df_pareto_11_2_with_score['Negative Comfort Days']

    # 1) Plot Pareto with MCDM
    weighted_scores_11_2 = df_pareto_11_2_with_score['Weighted_Score'].values
    plot_pareto_front_with_mcdm(
        pareto_df=df_pareto_11_2_with_score,
        weighted_scores=weighted_scores_11_2,
        best_asf=best_asf_11_2,
        best_pseudo=best_pseudo_11_2,
        tradeoff_solutions=tradeoff_11_2
    )

    # 2) Clustering
    df_pareto_11_2_with_score = cluster_scenarios(df_pareto_11_2_with_score, n_clusters=5)
    plot_clusters(df_pareto_11_2_with_score)

    # 3) Sensitivity
    perform_sensitivity_analysis(df_pareto_11_2_with_score)

    # 4) Correlation matrix
    plot_correlation_matrix(df_pareto_11_2_with_score)

    # 5) 3D scatter
    plot_3d_scatter(df_pareto_11_2_with_score)

    # 6) Parallel coordinates
    plot_parallel_coordinates(df_pareto_11_2_with_score)

    # 7) [Optional] density-based plot
    # plot_density_based_pareto(df_pareto_11_2_with_score,
    #                           weighted_scores_11_2,
    #                           title='Density-Based Pareto - Approach 11.2')
