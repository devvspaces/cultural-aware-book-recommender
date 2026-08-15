"""
Formal Multi-Split Evaluation & Statistical Significance Runner
===============================================================
Evaluates Content-Based, Surprise SVD++, Cultural FM v1, and Cultural FM v2
across 5 randomized splits. Computes Mean ± Std for RMSE, MAE, Precision@5,
Recall@5, F1@5, nDCG@5, Diversity (ILD), Novelty, and Coverage.

Also conducts:
- Cold-start history stratification (0, 1-3, 4-10, 10+ ratings)
- Paired Student's t-test and Wilcoxon signed-rank significance testing

Usage:
    ./venv-surprise/bin/python project/evaluation/formal_evaluation.py
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from scipy import stats

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.join(BASE_DIR, "project")
GOODREADS_DIR = os.path.join(BASE_DIR, "goodreads")
sys.path.append(PROJECT_DIR)

import goodreads_content_recommender as gcr
from collaborative_svd import SurpriseSVDpp
from cultural_aware_fm import CulturallyAwareFM
from cultural_aware_fm_v2 import CulturallyAwareFMv2

def load_book_id_map(file_path):
    print("Loading book ID map from book_id_map.csv...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing {file_path}")
    df = pd.read_csv(file_path)
    df['book_id'] = df['book_id'].astype(str)
    return dict(zip(df['book_id'], df['book_id_csv']))

def load_ratings(interactions_path, book_csv_ids_set, limit=35000, max_lines=4000000):
    print(f"Streaming ratings from {interactions_path}...")
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
    df_ratings = pd.DataFrame(ratings)
    print(f"Loaded {len(df_ratings):,} ratings (scanned {lines_scanned:,} lines).")
    return df_ratings

def compute_ndcg_at_k(recommended_indices, relevant_books, k=5):
    """Compute Normalized Discounted Cumulative Gain (nDCG@K)."""
    dcg = 0.0
    for i, idx in enumerate(recommended_indices[:k]):
        if idx in relevant_books:
            dcg += 1.0 / np.log2(i + 2) # i=0 -> log2(2)=1
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_books))))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_ranking_and_diversity(df_test, df_train, books, model_context, svd_model, fm_v1, fm_v2,
                                   user_profiles, book_vectors, user_train_history,
                                   is_cold_start=False, k=5):
    candidate_indices = df_test['book_idx'].unique()
    test_users = df_test['user_idx'].unique()
    
    metrics = {
        'content': {'prec': [], 'rec': [], 'f1': [], 'ndcg': [], 'ild': [], 'novelty': [], 'rec_books': set()},
        'svd': {'prec': [], 'rec': [], 'f1': [], 'ndcg': [], 'ild': [], 'novelty': [], 'rec_books': set()},
        'fm_v1': {'prec': [], 'rec': [], 'f1': [], 'ndcg': [], 'ild': [], 'novelty': [], 'rec_books': set()},
        'fm_v2': {'prec': [], 'rec': [], 'f1': [], 'ndcg': [], 'ild': [], 'novelty': [], 'rec_books': set()}
    }
    
    book_pop = df_train['book_idx'].value_counts().to_dict()
    total_ratings = len(df_train)
    features = model_context["tfidf_matrix"]
    
    for user_idx in test_users:
        user_test_ratings = df_test[df_test['user_idx'] == user_idx]
        relevant_books = set(user_test_ratings[user_test_ratings['rating'] >= 3.5]['book_idx'])
        if not relevant_books:
            continue
            
        orig_user_hof = user_profiles.get(user_idx, fm_v1.global_average_hofstede) if not is_cold_start else fm_v1.global_average_hofstede
        history = user_train_history.get(user_idx, []) if not is_cold_start else []
        
        scores = {'content': [], 'svd': [], 'fm_v1': [], 'fm_v2': []}
        for b_idx in candidate_indices:
            scores['content'].append((b_idx, gcr.predict_rating(history, b_idx, model_context)))
            scores['svd'].append((b_idx, svd_model.predict(user_idx, b_idx)))
            scores['fm_v1'].append((b_idx, fm_v1.predict(
                None if is_cold_start else user_idx, b_idx, orig_user_hof, book_vectors[b_idx])))
            scores['fm_v2'].append((b_idx, fm_v2.predict(
                None if is_cold_start else user_idx, b_idx, orig_user_hof, book_vectors[b_idx])))
            
        for name in ['content', 'svd', 'fm_v1', 'fm_v2']:
            rec_list = [b for b, s in sorted(scores[name], key=lambda x: x[1], reverse=True)[:k]]
            hits = len(set(rec_list) & relevant_books)
            p = hits / k
            r = hits / len(relevant_books)
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
            ndcg = compute_ndcg_at_k(rec_list, relevant_books, k=k)
            
            metrics[name]['prec'].append(p)
            metrics[name]['rec'].append(r)
            metrics[name]['f1'].append(f1)
            metrics[name]['ndcg'].append(ndcg)
            metrics[name]['rec_books'].update(rec_list)
            
            user_novs = [-np.log2(book_pop.get(b, 1) / total_ratings) for b in rec_list]
            metrics[name]['novelty'].append(np.mean(user_novs))
            
            if len(rec_list) > 1:
                dists = []
                for i in range(len(rec_list)):
                    for j in range(i+1, len(rec_list)):
                        sim = features[rec_list[i]].dot(features[rec_list[j]].T).toarray()[0, 0]
                        dists.append(1.0 - max(0.0, min(1.0, float(sim))))
                metrics[name]['ild'].append(np.mean(dists))
            else:
                metrics[name]['ild'].append(0.0)
                
    summary = {}
    for name in ['content', 'svd', 'fm_v1', 'fm_v2']:
        summary[name] = {
            'precision': float(np.mean(metrics[name]['prec'])),
            'recall': float(np.mean(metrics[name]['rec'])),
            'f1': float(np.mean(metrics[name]['f1'])),
            'ndcg': float(np.mean(metrics[name]['ndcg'])),
            'ild': float(np.mean(metrics[name]['ild'])),
            'novelty': float(np.mean(metrics[name]['novelty'])),
            'coverage': float(len(metrics[name]['rec_books']) / len(candidate_indices))
        }
    return summary

def run_formal_evaluation(seeds=[42, 101, 777, 999, 2024]):
    print("=" * 80)
    print("  FORMAL MULTI-SPLIT EVALUATION & STATISTICAL SIGNIFICANCE BENCHMARK")
    print("=" * 80)
    
    # Paths
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
    
    vectorizer, tfidf_matrix = gcr.fit_tfidf_recommender(books)
    model_context = {"is_semantic": False, "tfidf_matrix": tfidf_matrix, "vectorizer": vectorizer}
    
    models = ['content', 'svd', 'fm_v1', 'fm_v2']
    split_results = {m: {'active_rmse': [], 'active_mae': [], 'cold_rmse': [], 'cold_mae': [],
                        'active_p5': [], 'active_r5': [], 'active_f1': [], 'active_ndcg': [],
                        'active_ild': [], 'active_nov': [], 'active_cov': [],
                        'cold_p5': [], 'cold_r5': [], 'cold_f1': [], 'cold_ndcg': [],
                        'cold_ild': [], 'cold_nov': [], 'cold_cov': []} for m in models}
    
    # Store per-user absolute error arrays across seeds for paired significance tests
    user_errors = {'content': [], 'svd': [], 'fm_v1': [], 'fm_v2': []}
    
    # Stratified cold-start tracking: buckets: [0, 1-3, 4-10, 10+]
    stratified_results = {m: {'0': [], '1-3': [], '4-10': [], '10+': []} for m in models}
    
    # Run across 5 randomized seeds
    for split_idx, seed in enumerate(seeds):
        print(f"\n>>> Running Cross-Validation Split {split_idx + 1}/{len(seeds)} (Seed: {seed}) <<<")
        np.random.seed(seed)
        shuffled_users = np.random.permutation(unique_users)
        cold_users = set(shuffled_users[:int(len(shuffled_users) * 0.15)])
        
        train_ratings, test_ratings, cold_test_ratings = [], [], []
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
                cold_test_ratings.append(robj)
            else:
                if np.random.rand() < 0.8:
                    train_ratings.append(robj)
                else:
                    test_ratings.append(robj)
                    
        df_train = pd.DataFrame(train_ratings)
        df_test = pd.DataFrame(test_ratings)
        df_cold = pd.DataFrame(cold_test_ratings)
        
        user_train_history = {}
        for _, row in df_train.iterrows():
            u = int(row['user_idx'])
            if u not in user_train_history:
                user_train_history[u] = []
            user_train_history[u].append((int(row['book_idx']), float(row['rating'])))
            
        # Train FM v1
        fm_v1 = CulturallyAwareFM(num_factors=10, learning_rate=0.01, regularization=0.03, epochs=8)
        fm_v1.load_hofstede_csv(hofstede_path)
        user_profiles, book_vectors = fm_v1.build_user_cultural_profiles(df_train, books, book_csv_to_idx)
        fm_v1.fit(df_train, books, book_csv_to_idx, user_profiles, book_vectors)
        
        # Train FM v2
        fm_v2 = CulturallyAwareFMv2(num_factors=10, learning_rate=0.01, regularization=0.03, epochs=8)
        fm_v2.load_hofstede_csv(hofstede_path)
        fm_v2.fit(df_train, books, book_csv_to_idx, user_profiles, book_vectors)
        
        # Train SVD++
        svd_model = SurpriseSVDpp(n_factors=10, n_epochs=8, random_state=seed)
        svd_train_df = pd.DataFrame({
            'user_id': df_train['user_idx'],
            'book_id': df_train['book_idx'],
            'rating': df_train['rating']
        })
        svd_model.fit(svd_train_df)
        
        # Active User Rating Predictions
        preds_act = {m: [] for m in models}
        actuals_act = []
        for _, row in df_test.iterrows():
            u_idx = int(row['user_idx'])
            b_idx = int(row['book_idx'])
            r_act = float(row['rating'])
            actuals_act.append(r_act)
            
            u_hof = user_profiles.get(u_idx, fm_v1.global_average_hofstede)
            b_hof = book_vectors[b_idx]
            
            c_p = gcr.predict_rating(user_train_history.get(u_idx, []), b_idx, model_context)
            s_p = svd_model.predict(u_idx, b_idx)
            f1_p = fm_v1.predict(u_idx, b_idx, u_hof, b_hof)
            f2_p = fm_v2.predict(u_idx, b_idx, u_hof, b_hof)
            
            preds_act['content'].append(c_p)
            preds_act['svd'].append(s_p)
            preds_act['fm_v1'].append(f1_p)
            preds_act['fm_v2'].append(f2_p)
            
            user_errors['content'].append(abs(r_act - c_p))
            user_errors['svd'].append(abs(r_act - s_p))
            user_errors['fm_v1'].append(abs(r_act - f1_p))
            user_errors['fm_v2'].append(abs(r_act - f2_p))
            
            # History stratification
            hist_len = len(user_train_history.get(u_idx, []))
            bucket = '1-3' if hist_len <= 3 else ('4-10' if hist_len <= 10 else '10+')
            stratified_results['content'][bucket].append(abs(r_act - c_p))
            stratified_results['svd'][bucket].append(abs(r_act - s_p))
            stratified_results['fm_v1'][bucket].append(abs(r_act - f1_p))
            stratified_results['fm_v2'][bucket].append(abs(r_act - f2_p))
            
        actuals_act = np.array(actuals_act)
        for m in models:
            p_arr = np.array(preds_act[m])
            split_results[m]['active_rmse'].append(float(np.sqrt(np.mean((actuals_act - p_arr) ** 2))))
            split_results[m]['active_mae'].append(float(np.mean(np.abs(actuals_act - p_arr))))
            
        # Cold User Rating Predictions
        preds_cold = {m: [] for m in models}
        actuals_cold = []
        for _, row in df_cold.iterrows():
            u_idx = int(row['user_idx'])
            b_idx = int(row['book_idx'])
            r_act = float(row['rating'])
            actuals_cold.append(r_act)
            b_hof = book_vectors[b_idx]
            
            c_p = gcr.predict_rating([], b_idx, model_context)
            s_p = svd_model.predict(u_idx, b_idx)
            f1_p = fm_v1.predict(None, b_idx, fm_v1.global_average_hofstede, b_hof)
            f2_p = fm_v2.predict(None, b_idx, fm_v2.global_average_hofstede, b_hof)
            
            preds_cold['content'].append(c_p)
            preds_cold['svd'].append(s_p)
            preds_cold['fm_v1'].append(f1_p)
            preds_cold['fm_v2'].append(f2_p)
            
            stratified_results['content']['0'].append(abs(r_act - c_p))
            stratified_results['svd']['0'].append(abs(r_act - s_p))
            stratified_results['fm_v1']['0'].append(abs(r_act - f1_p))
            stratified_results['fm_v2']['0'].append(abs(r_act - f2_p))
            
        actuals_cold = np.array(actuals_cold)
        for m in models:
            p_arr = np.array(preds_cold[m])
            split_results[m]['cold_rmse'].append(float(np.sqrt(np.mean((actuals_cold - p_arr) ** 2))))
            split_results[m]['cold_mae'].append(float(np.mean(np.abs(actuals_cold - p_arr))))
            
        # Top-K Ranking & Diversity
        norm_rank = evaluate_ranking_and_diversity(df_test, df_train, books, model_context, svd_model, fm_v1, fm_v2,
                                                   user_profiles, book_vectors, user_train_history, is_cold_start=False)
        cold_rank = evaluate_ranking_and_diversity(df_cold, df_train, books, model_context, svd_model, fm_v1, fm_v2,
                                                   user_profiles, book_vectors, user_train_history, is_cold_start=True)
        
        for m in models:
            split_results[m]['active_p5'].append(norm_rank[m]['precision'])
            split_results[m]['active_r5'].append(norm_rank[m]['recall'])
            split_results[m]['active_f1'].append(norm_rank[m]['f1'])
            split_results[m]['active_ndcg'].append(norm_rank[m]['ndcg'])
            split_results[m]['active_ild'].append(norm_rank[m]['ild'])
            split_results[m]['active_nov'].append(norm_rank[m]['novelty'])
            split_results[m]['active_cov'].append(norm_rank[m]['coverage'])
            
            split_results[m]['cold_p5'].append(cold_rank[m]['precision'])
            split_results[m]['cold_r5'].append(cold_rank[m]['recall'])
            split_results[m]['cold_f1'].append(cold_rank[m]['f1'])
            split_results[m]['cold_ndcg'].append(cold_rank[m]['ndcg'])
            split_results[m]['cold_ild'].append(cold_rank[m]['ild'])
            split_results[m]['cold_nov'].append(cold_rank[m]['novelty'])
            split_results[m]['cold_cov'].append(cold_rank[m]['coverage'])
            
    # Compute Statistical Significance Tests (FM v2 vs SVD++, FM v2 vs FM v1, FM v2 vs Content)
    print("\n" + "=" * 80)
    print("  PAIRED STATISTICAL SIGNIFICANCE TESTS (on User Absolute Error)")
    print("=" * 80)
    
    sig_tests = {}
    for comp in ['svd', 'fm_v1', 'content']:
        t_stat, p_val_t = stats.ttest_rel(user_errors['fm_v2'], user_errors[comp])
        w_stat, p_val_w = stats.wilcoxon(user_errors['fm_v2'][:10000], user_errors[comp][:10000])
        sig_tests[comp] = {
            't_stat': float(t_stat),
            'p_value_t': float(p_val_t),
            'wilcoxon_stat': float(w_stat),
            'p_value_w': float(p_val_w),
            'is_significant': bool(p_val_t < 0.05)
        }
        print(f"  FM v2 vs {comp.upper():<10} -> Paired t-test: t={t_stat:.4f}, p={p_val_t:.4e} (p < 0.05: {p_val_t < 0.05})")
        print(f"                          -> Wilcoxon test: W={w_stat:.4f}, p={p_val_w:.4e}")
        
    # Aggregate Mean and Std
    aggregated = {}
    for m in models:
        aggregated[m] = {}
        for k, v in split_results[m].items():
            aggregated[m][k] = {
                'mean': float(np.mean(v)),
                'std': float(np.std(v))
            }
            
    # Stratified History Averages (MAE per bucket)
    stratified_mae = {}
    for m in models:
        stratified_mae[m] = {
            b: float(np.mean(stratified_results[m][b])) if stratified_results[m][b] else 0.0
            for b in ['0', '1-3', '4-10', '10+']
        }
        
    # Save Full JSON results
    output_json_path = os.path.join(PROJECT_DIR, "evaluation", "evaluation_results.json")
    save_payload = {
        'seeds': seeds,
        'aggregated_metrics': aggregated,
        'significance_tests': sig_tests,
        'stratified_history_mae': stratified_mae
    }
    with open(output_json_path, 'w') as f:
        json.dump(save_payload, f, indent=2)
    print(f"\n✓ Saved formal evaluation results to: {output_json_path}")
    
    # Print Formal Publication Table
    print("\n" + "=" * 120)
    print("               FORMAL 5-FOLD MULTI-SPLIT MODEL EVALUATION RESULTS (MEAN ± STD)")
    print("=" * 120)
    
    rows = [
        ["Active User RMSE (↓)", f"{aggregated['content']['active_rmse']['mean']:.4f} ± {aggregated['content']['active_rmse']['std']:.4f}", f"{aggregated['svd']['active_rmse']['mean']:.4f} ± {aggregated['svd']['active_rmse']['std']:.4f}", f"{aggregated['fm_v1']['active_rmse']['mean']:.4f} ± {aggregated['fm_v1']['active_rmse']['std']:.4f}", f"{aggregated['fm_v2']['active_rmse']['mean']:.4f} ± {aggregated['fm_v2']['active_rmse']['std']:.4f}"],
        ["Active User MAE (↓)", f"{aggregated['content']['active_mae']['mean']:.4f} ± {aggregated['content']['active_mae']['std']:.4f}", f"{aggregated['svd']['active_mae']['mean']:.4f} ± {aggregated['svd']['active_mae']['std']:.4f}", f"{aggregated['fm_v1']['active_mae']['mean']:.4f} ± {aggregated['fm_v1']['active_mae']['std']:.4f}", f"{aggregated['fm_v2']['active_mae']['mean']:.4f} ± {aggregated['fm_v2']['active_mae']['std']:.4f}"],
        ["Active Precision@5 (↑)", f"{aggregated['content']['active_p5']['mean']:.4f} ± {aggregated['content']['active_p5']['std']:.4f}", f"{aggregated['svd']['active_p5']['mean']:.4f} ± {aggregated['svd']['active_p5']['std']:.4f}", f"{aggregated['fm_v1']['active_p5']['mean']:.4f} ± {aggregated['fm_v1']['active_p5']['std']:.4f}", f"{aggregated['fm_v2']['active_p5']['mean']:.4f} ± {aggregated['fm_v2']['active_p5']['std']:.4f}"],
        ["Active Recall@5 (↑)", f"{aggregated['content']['active_r5']['mean']:.4f} ± {aggregated['content']['active_r5']['std']:.4f}", f"{aggregated['svd']['active_r5']['mean']:.4f} ± {aggregated['svd']['active_r5']['std']:.4f}", f"{aggregated['fm_v1']['active_r5']['mean']:.4f} ± {aggregated['fm_v1']['active_r5']['std']:.4f}", f"{aggregated['fm_v2']['active_r5']['mean']:.4f} ± {aggregated['fm_v2']['active_r5']['std']:.4f}"],
        ["Active F1@5 (↑)", f"{aggregated['content']['active_f1']['mean']:.4f} ± {aggregated['content']['active_f1']['std']:.4f}", f"{aggregated['svd']['active_f1']['mean']:.4f} ± {aggregated['svd']['active_f1']['std']:.4f}", f"{aggregated['fm_v1']['active_f1']['mean']:.4f} ± {aggregated['fm_v1']['active_f1']['std']:.4f}", f"{aggregated['fm_v2']['active_f1']['mean']:.4f} ± {aggregated['fm_v2']['active_f1']['std']:.4f}"],
        ["Active nDCG@5 (↑)", f"{aggregated['content']['active_ndcg']['mean']:.4f} ± {aggregated['content']['active_ndcg']['std']:.4f}", f"{aggregated['svd']['active_ndcg']['mean']:.4f} ± {aggregated['svd']['active_ndcg']['std']:.4f}", f"{aggregated['fm_v1']['active_ndcg']['mean']:.4f} ± {aggregated['fm_v1']['active_ndcg']['std']:.4f}", f"{aggregated['fm_v2']['active_ndcg']['mean']:.4f} ± {aggregated['fm_v2']['active_ndcg']['std']:.4f}"],
        ["Active Diversity (ILD) (↑)", f"{aggregated['content']['active_ild']['mean']:.4f} ± {aggregated['content']['active_ild']['std']:.4f}", f"{aggregated['svd']['active_ild']['mean']:.4f} ± {aggregated['svd']['active_ild']['std']:.4f}", f"{aggregated['fm_v1']['active_ild']['mean']:.4f} ± {aggregated['fm_v1']['active_ild']['std']:.4f}", f"{aggregated['fm_v2']['active_ild']['mean']:.4f} ± {aggregated['fm_v2']['active_ild']['std']:.4f}"],
        ["Active Novelty (↑)", f"{aggregated['content']['active_nov']['mean']:.4f} ± {aggregated['content']['active_nov']['std']:.4f}", f"{aggregated['svd']['active_nov']['mean']:.4f} ± {aggregated['svd']['active_nov']['std']:.4f}", f"{aggregated['fm_v1']['active_nov']['mean']:.4f} ± {aggregated['fm_v1']['active_nov']['std']:.4f}", f"{aggregated['fm_v2']['active_nov']['mean']:.4f} ± {aggregated['fm_v2']['active_nov']['std']:.4f}"],
        ["Active Coverage (↑)", f"{aggregated['content']['active_cov']['mean']*100:.1f}%", f"{aggregated['svd']['active_cov']['mean']*100:.1f}%", f"{aggregated['fm_v1']['active_cov']['mean']*100:.1f}%", f"{aggregated['fm_v2']['active_cov']['mean']*100:.1f}%"],
        ["-" * 26, "-" * 18, "-" * 18, "-" * 18, "-" * 18],
        ["Cold-Start RMSE (↓)", f"{aggregated['content']['cold_rmse']['mean']:.4f} ± {aggregated['content']['cold_rmse']['std']:.4f}", f"{aggregated['svd']['cold_rmse']['mean']:.4f} ± {aggregated['svd']['cold_rmse']['std']:.4f}", f"{aggregated['fm_v1']['cold_rmse']['mean']:.4f} ± {aggregated['fm_v1']['cold_rmse']['std']:.4f}", f"{aggregated['fm_v2']['cold_rmse']['mean']:.4f} ± {aggregated['fm_v2']['cold_rmse']['std']:.4f}"],
        ["Cold-Start MAE (↓)", f"{aggregated['content']['cold_mae']['mean']:.4f} ± {aggregated['content']['cold_mae']['std']:.4f}", f"{aggregated['svd']['cold_mae']['mean']:.4f} ± {aggregated['svd']['cold_mae']['std']:.4f}", f"{aggregated['fm_v1']['cold_mae']['mean']:.4f} ± {aggregated['fm_v1']['cold_mae']['std']:.4f}", f"{aggregated['fm_v2']['cold_mae']['mean']:.4f} ± {aggregated['fm_v2']['cold_mae']['std']:.4f}"],
        ["Cold Precision@5 (↑)", f"{aggregated['content']['cold_p5']['mean']:.4f} ± {aggregated['content']['cold_p5']['std']:.4f}", f"{aggregated['svd']['cold_p5']['mean']:.4f} ± {aggregated['svd']['cold_p5']['std']:.4f}", f"{aggregated['fm_v1']['cold_p5']['mean']:.4f} ± {aggregated['fm_v1']['cold_p5']['std']:.4f}", f"{aggregated['fm_v2']['cold_p5']['mean']:.4f} ± {aggregated['fm_v2']['cold_p5']['std']:.4f}"],
        ["Cold Recall@5 (↑)", f"{aggregated['content']['cold_r5']['mean']:.4f} ± {aggregated['content']['cold_r5']['std']:.4f}", f"{aggregated['svd']['cold_r5']['mean']:.4f} ± {aggregated['svd']['cold_r5']['std']:.4f}", f"{aggregated['fm_v1']['cold_r5']['mean']:.4f} ± {aggregated['fm_v1']['cold_r5']['std']:.4f}", f"{aggregated['fm_v2']['cold_r5']['mean']:.4f} ± {aggregated['fm_v2']['cold_r5']['std']:.4f}"],
        ["Cold F1@5 (↑)", f"{aggregated['content']['cold_f1']['mean']:.4f} ± {aggregated['content']['cold_f1']['std']:.4f}", f"{aggregated['svd']['cold_f1']['mean']:.4f} ± {aggregated['svd']['cold_f1']['std']:.4f}", f"{aggregated['fm_v1']['cold_f1']['mean']:.4f} ± {aggregated['fm_v1']['cold_f1']['std']:.4f}", f"{aggregated['fm_v2']['cold_f1']['mean']:.4f} ± {aggregated['fm_v2']['cold_f1']['std']:.4f}"],
        ["Cold nDCG@5 (↑)", f"{aggregated['content']['cold_ndcg']['mean']:.4f} ± {aggregated['content']['cold_ndcg']['std']:.4f}", f"{aggregated['svd']['cold_ndcg']['mean']:.4f} ± {aggregated['svd']['cold_ndcg']['std']:.4f}", f"{aggregated['fm_v1']['cold_ndcg']['mean']:.4f} ± {aggregated['fm_v1']['cold_ndcg']['std']:.4f}", f"{aggregated['fm_v2']['cold_ndcg']['mean']:.4f} ± {aggregated['fm_v2']['cold_ndcg']['std']:.4f}"],
        ["Cold Diversity (ILD) (↑)", f"{aggregated['content']['cold_ild']['mean']:.4f} ± {aggregated['content']['cold_ild']['std']:.4f}", f"{aggregated['svd']['cold_ild']['mean']:.4f} ± {aggregated['svd']['cold_ild']['std']:.4f}", f"{aggregated['fm_v1']['cold_ild']['mean']:.4f} ± {aggregated['fm_v1']['cold_ild']['std']:.4f}", f"{aggregated['fm_v2']['cold_ild']['mean']:.4f} ± {aggregated['fm_v2']['cold_ild']['std']:.4f}"],
        ["Cold Novelty (↑)", f"{aggregated['content']['cold_nov']['mean']:.4f} ± {aggregated['content']['cold_nov']['std']:.4f}", f"{aggregated['svd']['cold_nov']['mean']:.4f} ± {aggregated['svd']['cold_nov']['std']:.4f}", f"{aggregated['fm_v1']['cold_nov']['mean']:.4f} ± {aggregated['fm_v1']['cold_nov']['std']:.4f}", f"{aggregated['fm_v2']['cold_nov']['mean']:.4f} ± {aggregated['fm_v2']['cold_nov']['std']:.4f}"],
        ["Cold Coverage (↑)", f"{aggregated['content']['cold_cov']['mean']*100:.1f}%", f"{aggregated['svd']['cold_cov']['mean']*100:.1f}%", f"{aggregated['fm_v1']['cold_cov']['mean']*100:.1f}%", f"{aggregated['fm_v2']['cold_cov']['mean']*100:.1f}%"],
    ]
    
    headers = ["Metric", "Content-Based", "Surprise SVD++", "Cultural FM v1", "Cultural FM v2"]
    widths = [len(h) for h in headers]
    for r in rows:
        for idx, val in enumerate(r):
            widths[idx] = max(widths[idx], len(str(val)))
            
    header_str = " | ".join(f"{headers[i]:<{widths[i]}}" for i in range(len(headers)))
    sep_str = "-|-".join("-" * widths[i] for i in range(len(headers)))
    print(f"| {header_str} |")
    print(f"| {sep_str} |")
    for r in rows:
        r_str = " | ".join(f"{str(r[i]):<{widths[i]}}" for i in range(len(r)))
        print(f"| {r_str} |")
    print("=" * 120)
    
    # Print Stratified MAE Table
    print("\n" + "=" * 80)
    print("          STRATIFIED PERFORMANCE BY USER RATING HISTORY (MAE ↓)")
    print("=" * 80)
    print(f"| {'User History Bucket':<25} | {'Content-Based':<13} | {'Surprise SVD++':<14} | {'Cultural FM v1':<14} | {'Cultural FM v2':<14} |")
    print(f"| {'-'*25} | {'-'*13} | {'-'*14} | {'-'*14} | {'-'*14} |")
    print(f"| {'0 ratings (Pure Cold)':<25} | {stratified_mae['content']['0']:<13.4f} | {stratified_mae['svd']['0']:<14.4f} | {stratified_mae['fm_v1']['0']:<14.4f} | {stratified_mae['fm_v2']['0']:<14.4f} |")
    print(f"| {'1-3 ratings (Early Warm)':<25} | {stratified_mae['content']['1-3']:<13.4f} | {stratified_mae['svd']['1-3']:<14.4f} | {stratified_mae['fm_v1']['1-3']:<14.4f} | {stratified_mae['fm_v2']['1-3']:<14.4f} |")
    print(f"| {'4-10 ratings (Medium Warm)':<25} | {stratified_mae['content']['4-10']:<13.4f} | {stratified_mae['svd']['4-10']:<14.4f} | {stratified_mae['fm_v1']['4-10']:<14.4f} | {stratified_mae['fm_v2']['4-10']:<14.4f} |")
    print(f"| {'10+ ratings (Active Mature)':<25} | {stratified_mae['content']['10+']:<13.4f} | {stratified_mae['svd']['10+']:<14.4f} | {stratified_mae['fm_v1']['10+']:<14.4f} | {stratified_mae['fm_v2']['10+']:<14.4f} |")
    print("=" * 80)

if __name__ == "__main__":
    run_formal_evaluation()
