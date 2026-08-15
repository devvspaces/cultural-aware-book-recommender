"""
Benchmarking Visualizations Generator
====================================
Reads formal evaluation results and creates publication-ready comparative charts:
- Grouped bar charts for Active vs Cold-Start RMSE & MAE
- Precision, Recall & nDCG@5 bar charts
- 6-axis Radar / Spider chart comparing all models
- User History Stratification Line chart (MAE vs Number of User Ratings)
- Cultural Distance vs Predicted Rating Analysis
- Cross-Country Cultural Preference Heatmap

Usage:
    ./venv-surprise/bin/python project/evaluation/generate_benchmark_charts.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.join(BASE_DIR, "project")
CHARTS_DIR = os.path.join(PROJECT_DIR, "evaluation", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif'
})

MODEL_NAMES = {
    'content': 'Content-Based',
    'svd': 'Surprise SVD++',
    'fm_v1': 'Cultural FM v1',
    'fm_v2': 'Cultural FM v2'
}

COLORS = {
    'content': '#4c72b0',
    'svd': '#55a868',
    'fm_v1': '#c44e52',
    'fm_v2': '#8172b3'
}

def save_fig(fig, name):
    path = os.path.join(CHARTS_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved chart: {path}")

def plot_bar_metrics(data):
    agg = data['aggregated_metrics']
    models = ['content', 'svd', 'fm_v1', 'fm_v2']
    labels = [MODEL_NAMES[m] for m in models]
    
    # 1. Active vs Cold RMSE & MAE
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    # Active
    active_rmse = [agg[m]['active_rmse']['mean'] for m in models]
    active_rmse_err = [agg[m]['active_rmse']['std'] for m in models]
    active_mae = [agg[m]['active_mae']['mean'] for m in models]
    active_mae_err = [agg[m]['active_mae']['std'] for m in models]
    
    axes[0].bar(x - width/2, active_rmse, width, yerr=active_rmse_err, capsize=4, label='RMSE', color='#4c72b0')
    axes[0].bar(x + width/2, active_mae, width, yerr=active_mae_err, capsize=4, label='MAE', color='#55a868')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha='right')
    axes[0].set_ylabel('Error Score (Lower is Better)')
    axes[0].set_title('Active User Rating Prediction Error')
    axes[0].legend()
    axes[0].set_ylim(0.5, 1.2)
    
    # Cold-Start
    cold_rmse = [agg[m]['cold_rmse']['mean'] for m in models]
    cold_rmse_err = [agg[m]['cold_rmse']['std'] for m in models]
    cold_mae = [agg[m]['cold_mae']['mean'] for m in models]
    cold_mae_err = [agg[m]['cold_mae']['std'] for m in models]
    
    axes[1].bar(x - width/2, cold_rmse, width, yerr=cold_rmse_err, capsize=4, label='RMSE', color='#c44e52')
    axes[1].bar(x + width/2, cold_mae, width, yerr=cold_mae_err, capsize=4, label='MAE', color='#8172b3')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha='right')
    axes[1].set_ylabel('Error Score (Lower is Better)')
    axes[1].set_title('Cold-Start User Rating Prediction Error')
    axes[1].legend()
    axes[1].set_ylim(0.5, 1.2)
    
    plt.tight_layout()
    save_fig(fig, "3a_active_vs_cold_rmse_mae")

    # 2. Ranking Metrics: Precision@5, Recall@5, nDCG@5
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Active Ranking
    p5_act = [agg[m]['active_p5']['mean'] for m in models]
    r5_act = [agg[m]['active_r5']['mean'] for m in models]
    ndcg_act = [agg[m]['active_ndcg']['mean'] for m in models]
    
    w = 0.25
    axes[0].bar(x - w, p5_act, w, label='Precision@5', color='#3498db')
    axes[0].bar(x, r5_act, w, label='Recall@5', color='#e74c3c')
    axes[0].bar(x + w, ndcg_act, w, label='nDCG@5', color='#2ecc71')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha='right')
    axes[0].set_ylabel('Score (Higher is Better)')
    axes[0].set_title('Active Users: Top-5 Ranking Metrics')
    axes[0].legend()
    
    # Cold-Start Ranking
    p5_cold = [agg[m]['cold_p5']['mean'] for m in models]
    r5_cold = [agg[m]['cold_r5']['mean'] for m in models]
    ndcg_cold = [agg[m]['cold_ndcg']['mean'] for m in models]
    
    axes[1].bar(x - w, p5_cold, w, label='Precision@5', color='#3498db')
    axes[1].bar(x, r5_cold, w, label='Recall@5', color='#e74c3c')
    axes[1].bar(x + w, ndcg_cold, w, label='nDCG@5', color='#2ecc71')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha='right')
    axes[1].set_ylabel('Score (Higher is Better)')
    axes[1].set_title('Cold-Start Users: Top-5 Ranking Metrics')
    axes[1].legend()
    
    plt.tight_layout()
    save_fig(fig, "3b_ranking_metrics_ndcg")

def plot_radar_chart(data):
    agg = data['aggregated_metrics']
    models = ['content', 'svd', 'fm_v1', 'fm_v2']
    
    categories = ['Accuracy (1/MAE)', 'Precision@5', 'Recall@5', 'Diversity (ILD)', 'Novelty (Norm)', 'Catalog Coverage']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    
    for m in models:
        # Normalized values between 0.1 and 1.0 for radar visualization
        mae_score = min(1.0, max(0.1, 1.0 - (agg[m]['active_mae']['mean'] - 0.70) / 0.30))
        p5_score = min(1.0, max(0.1, agg[m]['active_p5']['mean'] / 0.030))
        r5_score = min(1.0, max(0.1, agg[m]['active_r5']['mean'] / 0.120))
        ild_score = agg[m]['active_ild']['mean']
        nov_score = min(1.0, max(0.1, agg[m]['active_nov']['mean'] / 11.0))
        cov_score = min(1.0, max(0.05, agg[m]['active_cov']['mean'] / 0.65))
        
        values = [mae_score, p5_score, r5_score, ild_score, nov_score, cov_score]
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=MODEL_NAMES[m], color=COLORS[m])
        ax.fill(angles, values, color=COLORS[m], alpha=0.15)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1.05)
    plt.title("Multi-Dimensional Model Benchmark Radar Comparison", size=14, y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    save_fig(fig, "3c_radar_comparison")

def plot_stratified_mae(data):
    strat = data['stratified_history_mae']
    models = ['content', 'svd', 'fm_v1', 'fm_v2']
    buckets = ['0', '1-3', '4-10', '10+']
    bucket_labels = ['0 (Pure Cold)', '1–3 (Early Warm)', '4–10 (Medium Warm)', '10+ (Active Mature)']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in models:
        values = [strat[m][b] for b in buckets]
        ax.plot(bucket_labels, values, marker='o', linewidth=2.5, markersize=8, label=MODEL_NAMES[m], color=COLORS[m])
        for x_idx, val in enumerate(values):
            ax.annotate(f"{val:.3f}", (x_idx, val), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
            
    ax.set_xlabel("User Rating History Depth")
    ax.set_ylabel("Mean Absolute Error (MAE) — Lower is Better")
    ax.set_title("Stratified Model Performance Across User History Stages")
    ax.legend()
    plt.tight_layout()
    save_fig(fig, "3d_stratified_history_mae")

def plot_cultural_impact():
    """Scatter plot showing how predicted rating responds to cultural distance in FM v2."""
    hofstede_path = os.path.join(PROJECT_DIR, "hofstede.csv")
    if not os.path.exists(hofstede_path):
        return
    df_hof = pd.read_csv(hofstede_path)
    df_hof.columns = [c.strip().lower() for c in df_hof.columns]
    dims = ['pdi', 'idv', 'mas', 'uai', 'lto', 'ivr']
    for d in dims:
        df_hof[d] = pd.to_numeric(df_hof[d], errors='coerce').fillna(df_hof[d].median())
        
    hof_map = {row['country'].strip().lower(): row[dims].values for _, row in df_hof.iterrows()}
    
    np.random.seed(42)
    # Simulate pairs across varied cultural distances
    distances = []
    cos_sims = []
    sample_countries = list(hof_map.keys())
    for _ in range(300):
        c1, c2 = np.random.choice(sample_countries, 2, replace=False)
        v1, v2 = hof_map[c1], hof_map[c2]
        d = np.linalg.norm(v1 - v2) / (100.0 * np.sqrt(6.0))
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        distances.append(d)
        cos_sims.append(cos)
        
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(distances, cos_sims, c=distances, cmap='plasma', alpha=0.75, edgecolors='grey', s=60)
    ax.set_xlabel("Normalized Euclidean Cultural Distance (0 = Identical, 1 = Max Distance)")
    ax.set_ylabel("Cosine Cultural Alignment")
    ax.set_title("Cultural Distance vs. Cultural Alignment Feature Distribution")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Euclidean Cultural Gap")
    plt.tight_layout()
    save_fig(fig, "3e_cultural_distance_vs_alignment")

def plot_cross_country_heatmap():
    """Heatmap showing cultural compatibility distance across major African and Global countries."""
    hofstede_path = os.path.join(PROJECT_DIR, "hofstede.csv")
    if not os.path.exists(hofstede_path):
        return
    df_hof = pd.read_csv(hofstede_path)
    df_hof.columns = [c.strip().lower() for c in df_hof.columns]
    dims = ['pdi', 'idv', 'mas', 'uai', 'lto', 'ivr']
    for d in dims:
        df_hof[d] = pd.to_numeric(df_hof[d], errors='coerce').fillna(df_hof[d].median())
        
    target_countries = ['nigeria', 'south africa', 'egypt', 'ghana', 'kenya', 
                        'united states', 'united kingdom', 'japan', 'germany', 'brazil']
    
    country_vectors = {}
    for _, row in df_hof.iterrows():
        c_name = row['country'].strip().lower()
        if c_name in target_countries:
            country_vectors[c_name.title()] = row[dims].values
            
    countries = list(country_vectors.keys())
    N = len(countries)
    matrix = np.zeros((N, N))
    
    for i, c1 in enumerate(countries):
        for j, c2 in enumerate(countries):
            v1 = country_vectors[c1]
            v2 = country_vectors[c2]
            dist = np.linalg.norm(v1 - v2) / (100.0 * np.sqrt(6.0))
            matrix[i, j] = dist
            
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlOrRd", 
                xticklabels=countries, yticklabels=countries, ax=ax, linewidths=0.5)
    ax.set_title("Cross-Country Cultural Distance Matrix (0 = Close, 1 = Far)")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_fig(fig, "3f_cross_country_cultural_distance_matrix")

def main():
    json_path = os.path.join(PROJECT_DIR, "evaluation", "evaluation_results.json")
    if not os.path.exists(json_path):
        print(f"Error: Results JSON not found at {json_path}. Run formal_evaluation.py first.")
        sys.exit(1)
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    print("Generating Phase 3 Benchmarking Visualizations...")
    plot_bar_metrics(data)
    plot_radar_chart(data)
    plot_stratified_mae(data)
    plot_cultural_impact()
    plot_cross_country_heatmap()
    print(f"\nAll benchmark charts generated successfully in {CHARTS_DIR}!")

if __name__ == "__main__":
    main()
