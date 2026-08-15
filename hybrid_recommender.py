"""
Switching-Weighted Hybrid Recommendation Engine
================================================
Combines Culturally Aware Factorization Machine (FM v2) with Collaborative SVD++.

Architecture:
- For users with history depth n < T (Cold & Warm-Start):
    Uses Cultural FM v2 with explicit cultural distances & user/book Hofstede alignment.
- For active mature users (n >= T):
    Blends predictions via calibrated weighting:
    y_hybrid = alpha * y_FM_v2 + (1 - alpha) * y_SVDpp

Supports:
- Dynamic real-time user profile updates upon receiving new ratings.
- Top-K recommendation generation with diversity and cultural alignment scoring.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cultural_aware_fm_v2 import CulturallyAwareFMv2, compute_cultural_alignment_features
from collaborative_svd import SurpriseSVDpp

class HybridRecommender:
    def __init__(self, alpha=0.55, threshold_t=5, fm_factors=10, svd_factors=10, epochs=8):
        """
        alpha: Weight assigned to FM v2 for active users (1 - alpha for SVD++).
        threshold_t: Interaction count threshold separating Cold/Warm from Active users.
        """
        self.alpha = alpha
        self.threshold_t = threshold_t
        
        # Sub-models
        self.fm_v2 = CulturallyAwareFMv2(num_factors=fm_factors, epochs=epochs)
        self.svd_model = SurpriseSVDpp(n_factors=svd_factors, n_epochs=epochs)
        
        # Catalogs and internal state
        self.books = []
        self.book_vectors = []
        self.user_profiles = {}
        self.user_history = {} # {user_id: [(book_idx, rating)]}
        self.book_id_to_idx = {}
        self.is_fitted = False
        
    def load_hofstede_csv(self, file_path):
        """Pass through Hofstede CSV loading to FM v2."""
        self.fm_v2.load_hofstede_csv(file_path)
        
    def fit(self, df_train, books, book_id_to_idx):
        """
        Fit both sub-models on the training dataset.
        df_train: DataFrame with ['user_id', 'book_id', 'user_idx', 'book_idx', 'rating']
        """
        print("\n" + "="*80)
        print("  FITTING HYBRID RECOMMENDER (FM v2 + SURPRISE SVD++)")
        print("="*80)
        start_time = time.time()
        
        self.books = books
        self.book_id_to_idx = book_id_to_idx
        
        # 1. Build User Cultural Profiles & Book Vectors
        self.user_profiles, self.book_vectors = self.fm_v2.build_user_cultural_profiles(
            df_train, books, book_id_to_idx
        )
        
        # 2. Build User Interaction History Store
        self.user_history = {}
        for _, row in df_train.iterrows():
            u = int(row['user_idx'])
            b = int(row['book_idx'])
            r = float(row['rating'])
            if u not in self.user_history:
                self.user_history[u] = []
            self.user_history[u].append((b, r))
            
        # 3. Train Cultural FM v2
        print("\n[Hybrid Step 1/2] Training Culturally Aware FM v2...")
        self.fm_v2.fit(df_train, books, book_id_to_idx, self.user_profiles, self.book_vectors)
        
        # 4. Train Surprise SVD++
        print("\n[Hybrid Step 2/2] Training Surprise SVD++...")
        svd_train_df = pd.DataFrame({
            'user_id': df_train['user_idx'],
            'book_id': df_train['book_idx'],
            'rating': df_train['rating']
        })
        self.svd_model.fit(svd_train_df)
        
        self.is_fitted = True
        print(f"\n✓ Hybrid Recommender training completed in {time.time() - start_time:.2f} seconds.")

    def predict(self, user_id, book_idx, user_hof=None):
        """
        Predict rating using the switching-weighted hybrid policy.
        """
        if not self.is_fitted:
            raise RuntimeError("HybridRecommender must be fitted before predict.")
            
        book_hof = self.book_vectors[book_idx]
        
        # Retrieve user cultural profile if not explicitly passed
        if user_hof is None:
            if user_id is not None and user_id in self.user_profiles:
                user_hof = self.user_profiles[user_id]
            else:
                user_hof = self.fm_v2.global_average_hofstede
                
        # Determine user history depth & integer index
        history_len = len(self.user_history.get(user_id, [])) if user_id is not None else 0
        user_int_idx = user_id if isinstance(user_id, (int, np.integer)) else None
        
        # Case A: Cold-Start or Early Warm-Start (n < T) -> FM v2 Dominance
        if user_id is None or history_len < self.threshold_t:
            return self.fm_v2.predict(user_int_idx, book_idx, user_hof, book_hof)
            
        # Case B: Active Mature User (n >= T) -> Weighted Blend
        pred_fm = self.fm_v2.predict(user_int_idx, book_idx, user_hof, book_hof)
        pred_svd = self.svd_model.predict(user_id, book_idx)
        
        y_hybrid = (self.alpha * pred_fm) + ((1.0 - self.alpha) * pred_svd)
        return min(5.0, max(1.0, y_hybrid))

    def recommend_top_k(self, user_id=None, candidate_indices=None, user_hof=None, k=10, filter_rated=True):
        """
        Generate ranked top-K recommendations for a user.
        Returns a list of dictionaries with full book metadata and scores.
        """
        if candidate_indices is None:
            candidate_indices = range(len(self.books))
            
        rated_books = set()
        if user_id is not None and filter_rated:
            rated_books = {b for b, _ in self.user_history.get(user_id, [])}
            
        if user_hof is None:
            if user_id is not None and user_id in self.user_profiles:
                user_hof = self.user_profiles[user_id]
            else:
                user_hof = self.fm_v2.global_average_hofstede
                
        scored = []
        for b_idx in candidate_indices:
            if b_idx in rated_books:
                continue
                
            pred_score = self.predict(user_id, b_idx, user_hof)
            b_hof = self.book_vectors[b_idx]
            
            # Compute cultural alignment similarity for UI explanation
            eucl_d = np.linalg.norm(user_hof - b_hof) / (100.0 * np.sqrt(6.0))
            cultural_alignment = max(0.0, min(1.0, 1.0 - eucl_d))
            
            # Retrieve authors and genres with fallback
            authors = self.books[b_idx].get('authors', self.books[b_idx].get('authors_names', ['Unknown Author']))
            genres = self.books[b_idx].get('genres', [])
            if not genres:
                genres = self.books[b_idx].get('tags', [])[:3]
                
            desc = self.books[b_idx].get('description', '')
            
            scored.append({
                'book_idx': b_idx,
                'book_id': self.books[b_idx].get('book_id', ''),
                'title': self.books[b_idx].get('title', 'Unknown Title'),
                'authors': authors,
                'genres': genres[:3],
                'description': desc[:180] + ('...' if len(desc) > 180 else '') if desc else 'No description available.',
                'image_url': self.books[b_idx].get('image_url', ''),
                'country_code': self.books[b_idx].get('country_code', ''),
                'language_code': self.books[b_idx].get('language_code', ''),
                'predicted_rating': round(float(pred_score), 2),
                'cultural_alignment': round(float(cultural_alignment * 100), 1)
            })
            
        scored.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return scored[:k]

    def add_user_rating(self, user_id, book_idx, rating):
        """
        Dynamically update user history and re-aggregate their Hofstede profile.
        Enables instant, real-time recommendation updates in the web UI.
        """
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        self.user_history[user_id].append((book_idx, float(rating)))
        
        # Re-compute user cultural profile as average of positive ratings (>= 3.0)
        pos_vectors = [
            self.book_vectors[b] for b, r in self.user_history[user_id] if r >= 3.0
        ]
        if not pos_vectors:
            pos_vectors = [self.book_vectors[b] for b, _ in self.user_history[user_id]]
            
        if pos_vectors:
            self.user_profiles[user_id] = np.mean(pos_vectors, axis=0)
        else:
            self.user_profiles[user_id] = self.fm_v2.global_average_hofstede.copy()
            
        return self.user_profiles[user_id]
