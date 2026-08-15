import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FORCE_TFIDF = True

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import goodreads_content_recommender as gcr
from collaborative_svd import SurpriseSVDpp
from cultural_aware_fm import CulturallyAwareFM
from cultural_aware_fm_v2 import CulturallyAwareFMv2

def load_book_id_map(file_path):
    print("Loading book ID map from book_id_map.csv...")
    start_time = time.time()
    if not os.path.exists(file_path):
        print(f"Error: Map file not found at {file_path}")
        sys.exit(1)
    df = pd.read_csv(file_path)
    df['book_id'] = df['book_id'].astype(str)
    book_id_str_to_csv = dict(zip(df['book_id'], df['book_id_csv']))
    print(f"Loaded {len(book_id_str_to_csv)} mappings in {time.time() - start_time:.2f} seconds.")
    return book_id_str_to_csv

def load_ratings(interactions_path, book_csv_ids_set, limit=40000, max_lines=4000000):
    print(f"Streaming ratings from {interactions_path}...")
    start_time = time.time()
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
    print(f"Loaded {len(df_ratings)} ratings in {time.time() - start_time:.2f} seconds (Scanned {lines_scanned} lines).")
    return df_ratings

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return rmse, mae

def evaluate_ranking_and_diversity(df_test, df_train, books, model_context, svd_model, fm_v1, fm_v2,
                                   user_profiles, book_vectors, user_train_history, 
                                   is_cold_start=False, k=5):
    candidate_indices = df_test['book_idx'].unique()
    test_users = df_test['user_idx'].unique()
    
    metrics = {
        'content': {'prec': [], 'rec': [], 'ild': [], 'novelty': [], 'rec_books': set()},
        'svd': {'prec': [], 'rec': [], 'ild': [], 'novelty': [], 'rec_books': set()},
        'fm_v1': {'prec': [], 'rec': [], 'ild': [], 'novelty': [], 'rec_books': set()},
        'fm_v2': {'prec': [], 'rec': [], 'ild': [], 'novelty': [], 'rec_books': set()}
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
        
        content_scores = []
        svd_scores = []
        fm_v1_scores = []
        fm_v2_scores = []
        
        history = user_train_history.get(user_idx, []) if not is_cold_start else []
        
        for b_idx in candidate_indices:
            c_score = gcr.predict_rating(history, b_idx, model_context)
            content_scores.append((b_idx, c_score))
            
            s_score = svd_model.predict(user_idx, b_idx)
            svd_scores.append((b_idx, s_score))
            
            f1_score = fm_v1.predict(
                user_id=None if is_cold_start else user_idx,
                book_idx=b_idx,
                user_hof=orig_user_hof,
                book_hof=book_vectors[b_idx]
            )
            fm_v1_scores.append((b_idx, f1_score))
            
            f2_score = fm_v2.predict(
                user_id=None if is_cold_start else user_idx,
                book_idx=b_idx,
                user_hof=orig_user_hof,
                book_hof=book_vectors[b_idx]
            )
            fm_v2_scores.append((b_idx, f2_score))
            
        rec_content = [b for b, s in sorted(content_scores, key=lambda x: x[1], reverse=True)[:k]]
        rec_svd = [b for b, s in sorted(svd_scores, key=lambda x: x[1], reverse=True)[:k]]
        rec_fm_v1 = [b for b, s in sorted(fm_v1_scores, key=lambda x: x[1], reverse=True)[:k]]
        rec_fm_v2 = [b for b, s in sorted(fm_v2_scores, key=lambda x: x[1], reverse=True)[:k]]
        
        for name, rec_list in [('content', rec_content), ('svd', rec_svd), ('fm_v1', rec_fm_v1), ('fm_v2', rec_fm_v2)]:
            hits = len(set(rec_list) & relevant_books)
            metrics[name]['prec'].append(hits / k)
            metrics[name]['rec'].append(hits / len(relevant_books))
            metrics[name]['rec_books'].update(rec_list)
            
            user_novs = []
            for b in rec_list:
                pop = book_pop.get(b, 1) / total_ratings
                user_novs.append(-np.log2(pop))
            metrics[name]['novelty'].append(np.mean(user_novs))
            
            if len(rec_list) > 1:
                dists = []
                for i in range(len(rec_list)):
                    for j in range(i+1, len(rec_list)):
                        idx_a = rec_list[i]
                        idx_b = rec_list[j]
                        sim = features[idx_a].dot(features[idx_b].T).toarray()[0, 0]
                        dists.append(1.0 - max(0.0, min(1.0, float(sim))))
                metrics[name]['ild'].append(np.mean(dists))
            else:
                metrics[name]['ild'].append(0.0)
                
    summary = {}
    for name in ['content', 'svd', 'fm_v1', 'fm_v2']:
        summary[name] = {
            'precision': np.mean(metrics[name]['prec']),
            'recall': np.mean(metrics[name]['rec']),
            'ild': np.mean(metrics[name]['ild']),
            'novelty': np.mean(metrics[name]['novelty']),
            'coverage': len(metrics[name]['rec_books']) / len(candidate_indices)
        }
    return summary

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    goodreads_dir = os.path.join(base_dir, "goodreads")
    project_dir = os.path.join(base_dir, "project")
    
    authors_path = os.path.join(goodreads_dir, "goodreads_book_authors.json")
    genres_path = os.path.join(goodreads_dir, "goodreads_book_genres_initial.json")
    books_path = os.path.join(goodreads_dir, "goodreads_books.json")
    book_id_map_path = os.path.join(goodreads_dir, "book_id_map.csv")
    interactions_path = os.path.join(goodreads_dir, "goodreads_interactions.csv")
    hofstede_path = os.path.join(project_dir, "hofstede.csv")
    
    # 1. Load book ID mappings
    book_id_str_to_csv = load_book_id_map(book_id_map_path)
    
    # 2. Load books metadata
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
    print(f"Retained {len(books)} mapped books for collaborative filtering.")
    
    # 3. Load interactions
    df_ratings = load_ratings(interactions_path, book_csv_ids_set, limit=30000)
    
    if len(df_ratings) < 100:
        print("Error: Too few ratings loaded.")
        return
        
    unique_users = df_ratings['user_id'].unique()
    user_to_idx = {orig_u: idx for idx, orig_u in enumerate(unique_users)}
    df_ratings['user_idx'] = df_ratings['user_id'].map(user_to_idx)
    
    book_csv_to_idx = {b['book_id_csv']: idx for idx, b in enumerate(books)}
    df_ratings['book_idx'] = df_ratings['book_id'].map(book_csv_to_idx)
    
    # Fit TF-IDF content representation
    vectorizer, tfidf_matrix = gcr.fit_tfidf_recommender(books)
    model_context = {
        "is_semantic": False,
        "tfidf_matrix": tfidf_matrix,
        "vectorizer": vectorizer
    }
        
    # 4. Standard / Cold Start Splits
    np.random.seed(42)
    shuffled_users = np.random.permutation(unique_users)
    cold_users = set(shuffled_users[:int(len(shuffled_users) * 0.15)])
    
    train_ratings = []
    test_ratings = []
    cold_test_ratings = []
    
    for idx, row in df_ratings.iterrows():
        orig_u = row['user_id']
        rating_obj = {
            'user_id': int(row['user_id']),
            'book_id': int(row['book_id']),
            'user_idx': int(row['user_idx']),
            'book_idx': int(row['book_idx']),
            'rating': float(row['rating'])
        }
        
        if orig_u in cold_users:
            cold_test_ratings.append(rating_obj)
        else:
            if np.random.rand() < 0.8:
                train_ratings.append(rating_obj)
            else:
                test_ratings.append(rating_obj)
                
    df_train = pd.DataFrame(train_ratings)
    df_test = pd.DataFrame(test_ratings)
    df_cold = pd.DataFrame(cold_test_ratings)
    
    print(f"Normal Train size: {len(df_train)}, Test size: {len(df_test)}, Cold-Start Test size: {len(df_cold)}")
    
    user_train_history = {}
    for idx, row in df_train.iterrows():
        u = int(row['user_idx'])
        b = int(row['book_idx'])
        r = float(row['rating'])
        if u not in user_train_history:
            user_train_history[u] = []
        user_train_history[u].append((b, r))
        
    # 5. Initialize & Train Models
    # A. Culturally Aware FM (v1 Baseline)
    print("\n--- Training FM v1 (Baseline) ---")
    fm_v1 = CulturallyAwareFM(num_factors=10, learning_rate=0.01, regularization=0.03, epochs=8)
    fm_v1.load_hofstede_csv(hofstede_path)
    user_profiles, book_vectors = fm_v1.build_user_cultural_profiles(df_train, books, book_csv_to_idx)
    fm_v1.fit(df_train, books, book_csv_to_idx, user_profiles, book_vectors)
    
    # B. Culturally Aware FM (v2 Explicit Distance)
    print("\n--- Training FM v2 (Explicit Cultural Distance Features) ---")
    fm_v2 = CulturallyAwareFMv2(num_factors=10, learning_rate=0.01, regularization=0.03, epochs=8)
    fm_v2.load_hofstede_csv(hofstede_path)
    fm_v2.fit(df_train, books, book_csv_to_idx, user_profiles, book_vectors)
    
    # C. Surprise SVD++
    print("\n--- Training Surprise SVD++ ---")
    svd_model = SurpriseSVDpp(n_factors=10, n_epochs=8)
    svd_train_df = pd.DataFrame({
        'user_id': df_train['user_idx'],
        'book_id': df_train['book_idx'],
        'rating': df_train['rating']
    })
    svd_model.fit(svd_train_df)
    
    # 6. Evaluation - Scenario A: Normal Users (RMSE / MAE)
    print("\nEvaluating Rating Prediction accuracy for active users...")
    preds = {'content': [], 'svd': [], 'fm_v1': [], 'fm_v2': []}
    actuals = []
    
    for idx, row in df_test.iterrows():
        u_idx = int(row['user_idx'])
        b_idx = int(row['book_idx'])
        r_actual = float(row['rating'])
        actuals.append(r_actual)
        
        preds['content'].append(gcr.predict_rating(user_train_history.get(u_idx, []), b_idx, model_context))
        preds['svd'].append(svd_model.predict(u_idx, b_idx))
        
        u_hof = user_profiles.get(u_idx, fm_v1.global_average_hofstede)
        b_hof = book_vectors[b_idx]
        preds['fm_v1'].append(fm_v1.predict(u_idx, b_idx, u_hof, b_hof))
        preds['fm_v2'].append(fm_v2.predict(u_idx, b_idx, u_hof, b_hof))
        
    rmse_c, mae_c = compute_metrics(actuals, preds['content'])
    rmse_s, mae_s = compute_metrics(actuals, preds['svd'])
    rmse_f1, mae_f1 = compute_metrics(actuals, preds['fm_v1'])
    rmse_f2, mae_f2 = compute_metrics(actuals, preds['fm_v2'])
    
    # 7. Evaluation - Scenario B: Cold-Start Users (RMSE / MAE)
    print("Evaluating Rating Prediction accuracy for Cold-Start users...")
    cold_preds = {'content': [], 'svd': [], 'fm_v1': [], 'fm_v2': []}
    cold_actuals = []
    
    for idx, row in df_cold.iterrows():
        u_idx = int(row['user_idx'])
        b_idx = int(row['book_idx'])
        r_actual = float(row['rating'])
        cold_actuals.append(r_actual)
        
        cold_preds['content'].append(gcr.predict_rating([], b_idx, model_context))
        cold_preds['svd'].append(svd_model.predict(u_idx, b_idx))
        
        b_hof = book_vectors[b_idx]
        cold_preds['fm_v1'].append(fm_v1.predict(None, b_idx, fm_v1.global_average_hofstede, b_hof))
        cold_preds['fm_v2'].append(fm_v2.predict(None, b_idx, fm_v2.global_average_hofstede, b_hof))
        
    rmse_cc, mae_cc = compute_metrics(cold_actuals, cold_preds['content'])
    rmse_cs, mae_cs = compute_metrics(cold_actuals, cold_preds['svd'])
    rmse_cf1, mae_cf1 = compute_metrics(cold_actuals, cold_preds['fm_v1'])
    rmse_cf2, mae_cf2 = compute_metrics(cold_actuals, cold_preds['fm_v2'])
    
    # 8. Evaluation - Ranking, Diversity, Novelty, Coverage
    print("\nRunning top-K ranking, diversity, and coverage evaluation...")
    normal_ranking = evaluate_ranking_and_diversity(
        df_test, df_train, books, model_context, svd_model, fm_v1, fm_v2, 
        user_profiles, book_vectors, user_train_history, is_cold_start=False
    )
    
    cold_ranking = evaluate_ranking_and_diversity(
        df_cold, df_train, books, model_context, svd_model, fm_v1, fm_v2, 
        user_profiles, book_vectors, user_train_history, is_cold_start=True
    )
    
    print("\n" + "="*110)
    print("                     GOODREADS MODEL EVALUATION COMPARISON REPORT (V1 vs V2)")
    print("="*110)
    
    table_data = [
        ["Active User RMSE", f"{rmse_c:.4f}", f"{rmse_s:.4f}", f"{rmse_f1:.4f}", f"{rmse_f2:.4f}"],
        ["Active User MAE", f"{mae_c:.4f}", f"{mae_s:.4f}", f"{mae_f1:.4f}", f"{mae_f2:.4f}"],
        ["Active Precision@5", f"{normal_ranking['content']['precision']:.4f}", f"{normal_ranking['svd']['precision']:.4f}", f"{normal_ranking['fm_v1']['precision']:.4f}", f"{normal_ranking['fm_v2']['precision']:.4f}"],
        ["Active Recall@5", f"{normal_ranking['content']['recall']:.4f}", f"{normal_ranking['svd']['recall']:.4f}", f"{normal_ranking['fm_v1']['recall']:.4f}", f"{normal_ranking['fm_v2']['recall']:.4f}"],
        ["Active Diversity (ILD)", f"{normal_ranking['content']['ild']:.4f}", f"{normal_ranking['svd']['ild']:.4f}", f"{normal_ranking['fm_v1']['ild']:.4f}", f"{normal_ranking['fm_v2']['ild']:.4f}"],
        ["Active Novelty", f"{normal_ranking['content']['novelty']:.4f}", f"{normal_ranking['svd']['novelty']:.4f}", f"{normal_ranking['fm_v1']['novelty']:.4f}", f"{normal_ranking['fm_v2']['novelty']:.4f}"],
        ["Active Coverage", f"{normal_ranking['content']['coverage']*100:.1f}%", f"{normal_ranking['svd']['coverage']*100:.1f}%", f"{normal_ranking['fm_v1']['coverage']*100:.1f}%", f"{normal_ranking['fm_v2']['coverage']*100:.1f}%"],
        ["-"*24, "-"*10, "-"*10, "-"*10, "-"*10],
        ["Cold-Start RMSE", f"{rmse_cc:.4f}", f"{rmse_cs:.4f}", f"{rmse_cf1:.4f}", f"{rmse_cf2:.4f}"],
        ["Cold-Start MAE", f"{mae_cc:.4f}", f"{mae_cs:.4f}", f"{mae_cf1:.4f}", f"{mae_cf2:.4f}"],
        ["Cold Precision@5", f"{cold_ranking['content']['precision']:.4f}", f"{cold_ranking['svd']['precision']:.4f}", f"{cold_ranking['fm_v1']['precision']:.4f}", f"{cold_ranking['fm_v2']['precision']:.4f}"],
        ["Cold Recall@5", f"{cold_ranking['content']['recall']:.4f}", f"{cold_ranking['svd']['recall']:.4f}", f"{cold_ranking['fm_v1']['recall']:.4f}", f"{cold_ranking['fm_v2']['recall']:.4f}"],
        ["Cold Diversity (ILD)", f"{cold_ranking['content']['ild']:.4f}", f"{cold_ranking['svd']['ild']:.4f}", f"{cold_ranking['fm_v1']['ild']:.4f}", f"{cold_ranking['fm_v2']['ild']:.4f}"],
        ["Cold Novelty", f"{cold_ranking['content']['novelty']:.4f}", f"{cold_ranking['svd']['novelty']:.4f}", f"{cold_ranking['fm_v1']['novelty']:.4f}", f"{cold_ranking['fm_v2']['novelty']:.4f}"],
        ["Cold Coverage", f"{cold_ranking['content']['coverage']*100:.1f}%", f"{cold_ranking['svd']['coverage']*100:.1f}%", f"{cold_ranking['fm_v1']['coverage']*100:.1f}%", f"{cold_ranking['fm_v2']['coverage']*100:.1f}%"],
    ]
    
    headers = ["Evaluation Metric", "Content-Based", "Surprise SVD++", "Cultural FM v1", "Cultural FM v2"]
    col_widths = [len(h) for h in headers]
    for row in table_data:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(val)))
            
    header_str = " | ".join(f"{str(val):<{col_widths[idx]}}" for idx, val in enumerate(headers))
    sep_str = "-|-".join("-" * col_widths[idx] for idx in range(len(headers)))
    
    print(f"| {header_str} |")
    print(f"| {sep_str} |")
    for row in table_data:
        row_str = " | ".join(f"{str(val):<{col_widths[idx]}}" for idx, val in enumerate(row))
        print(f"| {row_str} |")
    print("="*110)

if __name__ == "__main__":
    main()
