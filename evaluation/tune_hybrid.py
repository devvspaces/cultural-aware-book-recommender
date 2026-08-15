"""
Hybrid Engine Hyperparameter Tuning & Comparative Benchmark
============================================================
Performs grid-search over:
- Threshold T in {1, 3, 5, 8, 10}
- Alpha in {0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9}

Evaluates the tuned Hybrid Recommender vs. Standalone SVD++ and Standalone FM v2
on a held-out test split.

Usage:
    ./venv-surprise/bin/python project/evaluation/tune_hybrid.py
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.join(BASE_DIR, "project")
GOODREADS_DIR = os.path.join(BASE_DIR, "goodreads")
CHARTS_DIR = os.path.join(PROJECT_DIR, "evaluation", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)
sys.path.append(PROJECT_DIR)

import goodreads_content_recommender as gcr
from hybrid_recommender import HybridRecommender
from cultural_aware_fm_v2 import CulturallyAwareFMv2
from collaborative_svd import SurpriseSVDpp

def load_book_id_map(file_path):
    df = pd.read_csv(file_path)
    df['book_id'] = df['book_id'].astype(str)
    return dict(zip(df['book_id'], df['book_id_csv']))

def load_ratings(interactions_path, book_csv_ids_set, limit=35000, max_lines=4000000):
    ratings = []
    lines_scanned = 0
    for chunk in pd.read_csv(interactions_path, chunksize=100000):
        lines_scanned += len(chunk)
        filtered_chunk = chunk[(chunk['book_id'].isin(book_csv_ids_set)) & (chunk['rating'] > 0)]
        for _, row in filtered_chunk.iterrows():
            ratings.append({
                'user_id': int(row['user_id']),
                'book_id': int(row['book_id']),
                'rating': float(row['rating'])
            })
            if len(ratings) >= limit:
                break
        if len(ratings) >= limit or lines_scanned >= max_lines:
            break
    return pd.DataFrame(ratings)

def main():
    print("=" * 80)
    print("  HYBRID RECOMMENDER TUNING & BENCHMARK")
    print("=" * 80)
    
    authors_path = os.path.join(GOODREADS_DIR, "goodreads_book_authors.json")
    genres_path = os.path.join(GOODREADS_DIR, "goodreads_book_genres_initial.json")
    books_path = os.path.join(GOODREADS_DIR, "goodreads_books.json")
    book_id_map_path = os.path.join(GOODREADS_DIR, "book_id_map.csv")
    interactions_path = os.path.join(GOODREADS_DIR, "goodreads_interactions.csv")
    hofstede_path = os.path.join(PROJECT_DIR, "hofstede.csv")
    
    book_id_str_to_csv = load_book_id_map(book_id_map_path)
    authors_map = gcr.load_authors(authors_path)
    genres_map = gcr.load_genres(genres_path)
    
    limit_books = 15000
    books = gcr.load_books_dataset(books_path, authors_map, genres_map, limit=limit_books)
    
    filtered_books = []
    book_csv_ids_set = set()
    for b in books:
        s_id = b["book_id"]
        if s_id in book_id_str_to_csv:
            csv_id = book_id_str_to_csv[s_id]
            b["book_id_csv"] = csv_id
            filtered_books.append(b)
            book_csv_ids_set.add(csv_id)
    books = filtered_books
    print(f"Retained {len(books):,} mapped books.")
    
    df_ratings = load_ratings(interactions_path, book_csv_ids_set, limit=30000)
    unique_users = df_ratings['user_id'].unique()
    user_to_idx = {orig_u: idx for idx, orig_u in enumerate(unique_users)}
    df_ratings['user_idx'] = df_ratings['user_id'].map(user_to_idx)
    book_csv_to_idx = {b['book_id_csv']: idx for idx, b in enumerate(books)}
    df_ratings['book_idx'] = df_ratings['book_id'].map(book_csv_to_idx)
    
    # 3-Way Split: Train (70%), Val (15%), Test (15%)
    np.random.seed(42)
    shuffled_users = np.random.permutation(unique_users)
    n_total = len(shuffled_users)
    cold_users = set(shuffled_users[:int(n_total * 0.15)])
    
    train_ratings, val_ratings, test_ratings, cold_ratings = [], [], [], []
    for _, row in df_ratings.iterrows():
        orig_u = row['user_id']
        robj = {
            'user_id': int(row['user_id']),
            'book_id': int(row['book_id']),
            'user_idx': int(row['user_idx']),
            'book_idx': int(row['book_idx']),
            'rating': float(row['rating'])
        }
        if orig_u in cold_users:
            cold_ratings.append(robj)
        else:
            rnd = np.random.rand()
            if rnd < 0.70:
                train_ratings.append(robj)
            elif rnd < 0.85:
                val_ratings.append(robj)
            else:
                test_ratings.append(robj)
                
    df_train = pd.DataFrame(train_ratings)
    df_val = pd.DataFrame(val_ratings)
    df_test = pd.DataFrame(test_ratings)
    df_cold = pd.DataFrame(cold_ratings)
    
    print(f"Train size: {len(df_train):,}, Val size: {len(df_val):,}, Test size: {len(df_test):,}, Cold size: {len(df_cold):,}")
    
    # Fit base models once for validation grid search
    hybrid = HybridRecommender(alpha=0.5, threshold_t=5, fm_factors=10, svd_factors=10, epochs=8)
    hybrid.load_hofstede_csv(hofstede_path)
    hybrid.fit(df_train, books, book_csv_to_idx)
    
    # Grid Search over T and Alpha on Validation Split
    print("\n--- Running Grid Search on Validation Split ---")
    thresholds = [1, 3, 5, 8, 10]
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]
    
    best_mae = float('inf')
    best_rmse = float('inf')
    best_params = (0.5, 5)
    
    grid_results = []
    
    # Precompute raw validation predictions from submodels for fast tuning
    val_fm_preds = []
    val_svd_preds = []
    val_actuals = []
    val_user_hist_lens = []
    
    for _, row in df_val.iterrows():
        u_idx = int(row['user_idx'])
        b_idx = int(row['book_idx'])
        r_act = float(row['rating'])
        val_actuals.append(r_act)
        
        u_hof = hybrid.user_profiles.get(u_idx, hybrid.fm_v2.global_average_hofstede)
        b_hof = hybrid.book_vectors[b_idx]
        
        val_fm_preds.append(hybrid.fm_v2.predict(u_idx, b_idx, u_hof, b_hof))
        val_svd_preds.append(hybrid.svd_model.predict(u_idx, b_idx))
        val_user_hist_lens.append(len(hybrid.user_history.get(u_idx, [])))
        
    val_actuals = np.array(val_actuals)
    val_fm_preds = np.array(val_fm_preds)
    val_svd_preds = np.array(val_svd_preds)
    val_user_hist_lens = np.array(val_user_hist_lens)
    
    for T in thresholds:
        for a in alphas:
            hybrid_preds = np.where(
                val_user_hist_lens < T,
                val_fm_preds,
                (a * val_fm_preds) + ((1.0 - a) * val_svd_preds)
            )
            hybrid_preds = np.clip(hybrid_preds, 1.0, 5.0)
            
            mae = float(np.mean(np.abs(val_actuals - hybrid_preds)))
            rmse = float(np.sqrt(np.mean((val_actuals - hybrid_preds) ** 2)))
            grid_results.append({'T': T, 'alpha': a, 'mae': mae, 'rmse': rmse})
            
            if mae < best_mae:
                best_mae = mae
                best_rmse = rmse
                best_params = (a, T)
                
    opt_alpha, opt_T = best_params
    print(f"\n★ OPTIMAL HYPERPARAMETERS: Threshold T = {opt_T}, Alpha = {opt_alpha:.2f}")
    print(f"  Validation MAE: {best_mae:.4f}, Validation RMSE: {best_rmse:.4f}")
    
    # Configure Hybrid with Optimal Parameters
    hybrid.alpha = opt_alpha
    hybrid.threshold_t = opt_T
    
    # Evaluate on Held-Out Test Split: Standalone SVD++ vs Standalone FM v2 vs Tuned Hybrid
    print("\n--- Final Held-Out Test Set Benchmark ---")
    
    def eval_split(df_eval, is_cold=False):
        actuals = []
        preds_svd = []
        preds_fm = []
        preds_hyb = []
        
        for _, row in df_eval.iterrows():
            u_idx = int(row['user_idx']) if not is_cold else None
            b_idx = int(row['book_idx'])
            r_act = float(row['rating'])
            actuals.append(r_act)
            
            u_hof = hybrid.user_profiles.get(u_idx, hybrid.fm_v2.global_average_hofstede) if not is_cold else hybrid.fm_v2.global_average_hofstede
            b_hof = hybrid.book_vectors[b_idx]
            
            p_svd = hybrid.svd_model.predict(u_idx if u_idx is not None else 0, b_idx)
            p_fm = hybrid.fm_v2.predict(u_idx, b_idx, u_hof, b_hof)
            p_hyb = hybrid.predict(u_idx, b_idx, u_hof)
            
            preds_svd.append(p_svd)
            preds_fm.append(p_fm)
            preds_hyb.append(p_hyb)
            
        actuals = np.array(actuals)
        res = {}
        for name, p_list in [('svd', preds_svd), ('fm_v2', preds_fm), ('hybrid', preds_hyb)]:
            p_arr = np.array(p_list)
            res[name] = {
                'rmse': float(np.sqrt(np.mean((actuals - p_arr) ** 2))),
                'mae': float(np.mean(np.abs(actuals - p_arr)))
            }
        return res

    active_test_res = eval_split(df_test, is_cold=False)
    cold_test_res = eval_split(df_cold, is_cold=True)
    
    # Print Comparison Table
    print("\n" + "=" * 95)
    print(f"       HELD-OUT TEST SET COMPARISON: STANDALONE vs TUNED HYBRID (T={opt_T}, α={opt_alpha:.2f})")
    print("=" * 95)
    print(f"| {'Evaluation Metric':<26} | {'Surprise SVD++':<16} | {'Cultural FM v2':<16} | {'Hybrid Recommender':<20} |")
    print(f"| {'-'*26} | {'-'*16} | {'-'*16} | {'-'*20} |")
    print(f"| {'Active User RMSE (↓)':<26} | {active_test_res['svd']['rmse']:<16.4f} | {active_test_res['fm_v2']['rmse']:<16.4f} | {active_test_res['hybrid']['rmse']:<20.4f} |")
    print(f"| {'Active User MAE (↓)':<26} | {active_test_res['svd']['mae']:<16.4f} | {active_test_res['fm_v2']['mae']:<16.4f} | {active_test_res['hybrid']['mae']:<20.4f} |")
    print(f"| {'Cold-Start User RMSE (↓)':<26} | {cold_test_res['svd']['rmse']:<16.4f} | {cold_test_res['fm_v2']['rmse']:<16.4f} | {cold_test_res['hybrid']['rmse']:<20.4f} |")
    print(f"| {'Cold-Start User MAE (↓)':<26} | {cold_test_res['svd']['mae']:<16.4f} | {cold_test_res['fm_v2']['mae']:<16.4f} | {cold_test_res['hybrid']['mae']:<20.4f} |")
    print("=" * 95)
    
    # Save Hybrid Test JSON
    save_payload = {
        'optimal_params': {'threshold_T': opt_T, 'alpha': opt_alpha},
        'grid_search_summary': grid_results,
        'active_test_results': active_test_res,
        'cold_test_results': cold_test_res
    }
    out_json = os.path.join(PROJECT_DIR, "evaluation", "hybrid_evaluation_results.json")
    with open(out_json, 'w') as f:
        json.dump(save_payload, f, indent=2)
    print(f"\n✓ Saved hybrid evaluation results to: {out_json}")
    
    # Plot Grid Search Surface & Comparative Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Parameter Grid Heatmap
    df_grid = pd.DataFrame(grid_results)
    pivot_mae = df_grid.pivot(index='T', columns='alpha', values='mae')
    sns.heatmap(pivot_mae, annot=True, fmt=".4f", cmap="YlGnBu_r", ax=axes[0])
    axes[0].set_title(f"Validation MAE Grid (Min: T={opt_T}, α={opt_alpha:.2f})")
    axes[0].set_xlabel("Blend Weight α (FM v2 Weight)")
    axes[0].set_ylabel("Switching Threshold T")
    
    # 2. Test Set Comparison Bar Chart
    models = ['Surprise SVD++', 'Cultural FM v2', 'Hybrid Engine']
    act_mae = [active_test_res['svd']['mae'], active_test_res['fm_v2']['mae'], active_test_res['hybrid']['mae']]
    cold_mae = [cold_test_res['svd']['mae'], cold_test_res['fm_v2']['mae'], cold_test_res['hybrid']['mae']]
    
    x = np.arange(len(models))
    w = 0.35
    axes[1].bar(x - w/2, act_mae, w, label='Active MAE', color='#3498db')
    axes[1].bar(x + w/2, cold_mae, w, label='Cold-Start MAE', color='#e67e22')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=10)
    axes[1].set_ylabel("MAE (Lower is Better)")
    axes[1].set_title("Test Set Error Comparison (Standalone vs Hybrid)")
    axes[1].set_ylim(0.65, 0.85)
    axes[1].legend()
    for idx in range(len(models)):
        axes[1].text(x[idx] - w/2, act_mae[idx] + 0.005, f"{act_mae[idx]:.3f}", ha='center', fontsize=9)
        axes[1].text(x[idx] + w/2, cold_mae[idx] + 0.005, f"{cold_mae[idx]:.3f}", ha='center', fontsize=9)
        
    plt.tight_layout()
    chart_path = os.path.join(CHARTS_DIR, "4a_hybrid_tuning_and_comparison.png")
    fig.savefig(chart_path, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved chart: {chart_path}")

if __name__ == "__main__":
    main()
