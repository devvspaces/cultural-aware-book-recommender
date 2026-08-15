"""
Model Serialization & Artifact Exporter
=======================================
Trains and serializes:
1. Cultural FM v2 weights (w0, w, V, Hofstede vectors) to .npz and .json
2. SVD++ model via Surprise dump
3. Lightweight preprocessed book metadata catalog (books_catalog.json)
4. Recommender configuration parameters (recommender_config.json)

Usage:
    ./venv-surprise/bin/python project/scripts/export_models.py
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from surprise import dump

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.join(BASE_DIR, "project")
GOODREADS_DIR = os.path.join(BASE_DIR, "goodreads")
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
sys.path.append(PROJECT_DIR)

import goodreads_content_recommender as gcr
from hybrid_recommender import HybridRecommender

def main():
    print("=" * 80)
    print("  SERIALIZING RECOMMENDER ARTIFACTS FOR CLOUD DEPLOYMENT")
    print("=" * 80)
    start_time = time.time()
    
    authors_path = os.path.join(GOODREADS_DIR, "goodreads_book_authors.json")
    genres_path = os.path.join(GOODREADS_DIR, "goodreads_book_genres_initial.json")
    books_path = os.path.join(GOODREADS_DIR, "goodreads_books.json")
    book_id_map_path = os.path.join(GOODREADS_DIR, "book_id_map.csv")
    interactions_path = os.path.join(GOODREADS_DIR, "goodreads_interactions.csv")
    hofstede_path = os.path.join(PROJECT_DIR, "hofstede.csv")
    
    # 1. Load book ID mappings
    df_map = pd.read_csv(book_id_map_path)
    df_map['book_id'] = df_map['book_id'].astype(str)
    book_id_str_to_csv = dict(zip(df_map['book_id'], df_map['book_id_csv']))
    
    # 2. Load books
    authors_map = gcr.load_authors(authors_path)
    genres_map = gcr.load_genres(genres_path)
    raw_books = gcr.load_books_dataset(books_path, authors_map, genres_map, limit=50000)
    
    filtered_books = []
    book_csv_ids_set = set()
    for b in raw_books:
        s_id = b["book_id"]
        if s_id in book_id_str_to_csv:
            csv_id = book_id_str_to_csv[s_id]
            b["book_id_csv"] = csv_id
            filtered_books.append(b)
            book_csv_ids_set.add(csv_id)
            
    books = filtered_books
    book_csv_to_idx = {b['book_id_csv']: idx for idx, b in enumerate(books)}
    print(f"Catalog size: {len(books):,} books.")
    
    # 3. Load interactions
    ratings = []
    for chunk in pd.read_csv(interactions_path, chunksize=100000):
        filtered = chunk[(chunk['book_id'].isin(book_csv_ids_set)) & (chunk['rating'] > 0)]
        for _, row in filtered.iterrows():
            ratings.append({
                'user_id': int(row['user_id']),
                'book_id': int(row['book_id']),
                'rating': float(row['rating'])
            })
            if len(ratings) >= 40000:
                break
        if len(ratings) >= 40000:
            break
            
    df_ratings = pd.DataFrame(ratings)
    unique_users = df_ratings['user_id'].unique()
    user_to_idx = {orig_u: idx for idx, orig_u in enumerate(unique_users)}
    df_ratings['user_idx'] = df_ratings['user_id'].map(user_to_idx)
    df_ratings['book_idx'] = df_ratings['book_id'].map(book_csv_to_idx)
    
    # 4. Train Hybrid Recommender
    hybrid = HybridRecommender(alpha=0.80, threshold_t=1, fm_factors=10, svd_factors=10, epochs=8)
    hybrid.load_hofstede_csv(hofstede_path)
    hybrid.fit(df_ratings, books, book_csv_to_idx)
    
    # 5. Export FM v2 weights
    fm_npz_path = os.path.join(ARTIFACTS_DIR, "fm_v2_weights.npz")
    np.savez_compressed(
        fm_npz_path,
        w0=hybrid.fm_v2.w0,
        w=hybrid.fm_v2.w,
        V=hybrid.fm_v2.V,
        global_average_hofstede=hybrid.fm_v2.global_average_hofstede
    )
    print(f"✓ Saved FM v2 weights to: {fm_npz_path}")
    
    # 6. Export SVD++ model
    svd_path = os.path.join(ARTIFACTS_DIR, "svd_model.pkl")
    dump.dump(svd_path, algo=hybrid.svd_model.model)
    print(f"✓ Saved SVD++ model to: {svd_path}")
    
    # 7. Export Book Catalog JSON
    clean_catalog = []
    for idx, b in enumerate(books):
        b_genres = b.get("genres", [])
        if not b_genres:
            b_genres = b.get("tags", [])[:3]
            
        clean_catalog.append({
            "idx": idx,
            "book_id": b.get("book_id", ""),
            "title": b.get("title", ""),
            "authors": b.get("authors", b.get("authors_names", ["Unknown Author"])),
            "genres": b_genres[:3],
            "image_url": b.get("image_url", ""),
            "isbn": b.get("isbn", ""),
            "description": b.get("description", "")[:280] if b.get("description") else "",
            "average_rating": float(b.get("average_rating", 3.5)),
            "country_code": b.get("country_code", ""),
            "language_code": b.get("language_code", "")
        })
        
    catalog_path = os.path.join(ARTIFACTS_DIR, "books_catalog.json")
    with open(catalog_path, 'w', encoding='utf-8') as f:
        json.dump(clean_catalog, f, indent=2)
    print(f"✓ Saved clean book catalog to: {catalog_path}")
    
    # 8. Export Metadata & Config
    config_path = os.path.join(ARTIFACTS_DIR, "recommender_config.json")
    with open(config_path, 'w') as f:
        json.dump({
            "alpha": hybrid.alpha,
            "threshold_T": hybrid.threshold_t,
            "num_factors": hybrid.fm_v2.k,
            "num_users": hybrid.fm_v2.num_users,
            "num_books": hybrid.fm_v2.num_books,
            "num_continuous": hybrid.fm_v2.num_continuous,
            "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    print(f"✓ Saved deployment config to: {config_path}")
    
    print(f"\nAll artifacts exported successfully in {time.time() - start_time:.2f} seconds!")

if __name__ == "__main__":
    main()
