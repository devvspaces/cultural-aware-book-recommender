import os
import time
import numpy as np
import pandas as pd

ISO2_TO_COUNTRY = {
    'us': 'united states', 'gb': 'united kingdom', 'jp': 'japan', 'cn': 'china', 
    'de': 'germany', 'fr': 'france', 'in': 'india', 'mx': 'mexico', 
    'ng': 'nigeria', 'ca': 'canada', 'au': 'australia', 'br': 'brazil', 
    'ru': 'russia', 'za': 'south africa', 'es': 'spain', 'it': 'italy', 
    'nl': 'netherlands', 'se': 'sweden', 'no': 'norway', 'dk': 'denmark', 
    'fi': 'finland', 'ie': 'ireland', 'nz': 'new zealand', 'kr': 'south korea',
    'sg': 'singapore', 'hk': 'hong kong', 'tw': 'taiwan', 'eg': 'egypt',
    'sa': 'saudi arabia', 'tr': 'turkey', 'pk': 'pakistan', 'id': 'indonesia',
    'my': 'malaysia', 'ph': 'philippines', 'th': 'thailand', 'vn': 'vietnam',
    'ua': 'ukraine', 'pl': 'poland', 'ro': 'romania', 'gr': 'greece',
    'pt': 'portugal', 'be': 'belgium', 'ch': 'switzerland', 'at': 'austria',
    'cl': 'chile', 'co': 'colombia', 'pe': 'peru', 've': 'venezuela',
    'ar': 'argentina'
}

def compute_cultural_alignment_features(u_hof, b_hof):
    """
    Compute 20 continuous cultural alignment & distance features:
    - 6 User Hofstede scores (normalized 0-1)
    - 6 Book Hofstede scores (normalized 0-1)
    - 1 Normalized Euclidean Cultural Distance (0-1)
    - 1 Cosine Cultural Similarity (0-1)
    - 6 Dimension-Wise Absolute Differences (0-1)
    """
    u_norm = u_hof / 100.0
    b_norm = b_hof / 100.0
    
    # 1. Normalized Euclidean Distance: ||u - b|| / (100 * sqrt(6))
    eucl_dist = np.linalg.norm(u_hof - b_hof) / (100.0 * np.sqrt(6.0))
    
    # 2. Cosine Cultural Similarity
    norm_u = np.linalg.norm(u_hof)
    norm_b = np.linalg.norm(b_hof)
    if norm_u > 1e-6 and norm_b > 1e-6:
        cos_sim = float(np.dot(u_hof, b_hof) / (norm_u * norm_b))
    else:
        cos_sim = 0.5
        
    # 3. Dimension-Wise Absolute Gaps
    abs_gaps = np.abs(u_hof - b_hof) / 100.0
    
    return np.concatenate([u_norm, b_norm, [eucl_dist, cos_sim], abs_gaps])

class CulturallyAwareFMv2:
    def __init__(self, num_factors=10, learning_rate=0.01, regularization=0.03, epochs=8):
        self.k = num_factors
        self.lr = learning_rate
        self.reg = regularization
        self.epochs = epochs
        
        self.hofstede_map = {}
        self.global_average_hofstede = np.array([50.0] * 6)
        
        # Model weights
        self.w0 = 0.0
        self.w = None
        self.V = None
        
        # Dimensions
        self.num_users = 0
        self.num_books = 0
        self.num_continuous = 20
        self.hofstede_offset = 0
        self.total_features = 0
        self.is_fitted = False
        self.d = 0 # total features dimension
        
    def load_hofstede_csv(self, file_path):
        """Load and clean Hofstede CSV data, imputing missing values with column medians."""
        print("Loading Hofstede scores from CSV for FM v2...")
        if not os.path.exists(file_path):
            print(f"Warning: Hofstede CSV not found at {file_path}. Using default global averages.")
            return
            
        df = pd.read_csv(file_path)
        df.columns = [c.strip().lower() for c in df.columns]
        dimensions = ['pdi', 'idv', 'mas', 'uai', 'lto', 'ivr']
        
        for dim in dimensions:
            df[dim] = pd.to_numeric(df[dim], errors='coerce')
            
        medians = df[dimensions].median()
        df[dimensions] = df[dimensions].fillna(medians)
        
        self.hofstede_map = {}
        for _, row in df.iterrows():
            country = str(row['country']).strip().lower()
            vector = np.array([row[d] for d in dimensions], dtype=float)
            self.hofstede_map[country] = vector
            
        if self.hofstede_map:
            self.global_average_hofstede = np.mean(list(self.hofstede_map.values()), axis=0)
        print(f"FM v2 loaded {len(self.hofstede_map)} country profiles.")

    def get_book_hofstede(self, book):
        """Retrieve the Hofstede vector of a book based on its country or language code."""
        cc = str(book.get("country_code", "")).strip().lower()
        country_name = ISO2_TO_COUNTRY.get(cc)
        if country_name and country_name in self.hofstede_map:
            return self.hofstede_map[country_name]
            
        if cc in self.hofstede_map:
            return self.hofstede_map[cc]
            
        lang = str(book.get("language_code", "")).strip().lower()
        if 'spa' in lang:
            return self.hofstede_map.get('spain', self.global_average_hofstede)
        elif 'ger' in lang or 'deu' in lang:
            return self.hofstede_map.get('germany', self.global_average_hofstede)
        elif 'fre' in lang or 'fra' in lang:
            return self.hofstede_map.get('france', self.global_average_hofstede)
        elif 'jpn' in lang:
            return self.hofstede_map.get('japan', self.global_average_hofstede)
        elif 'rus' in lang:
            return self.hofstede_map.get('russia', self.global_average_hofstede)
        elif 'zho' in lang or 'chi' in lang:
            return self.hofstede_map.get('china', self.global_average_hofstede)
        elif 'por' in lang:
            return self.hofstede_map.get('brazil', self.global_average_hofstede)
            
        return self.global_average_hofstede

    def build_user_cultural_profiles(self, df_train, books, book_id_to_idx):
        """
        Compute bottom-up cultural vector profiles for historical users in the training set
        as the average Hofstede vector of books they rated >= 3.0.
        """
        print("Propagating book metadata to build user cultural profiles (v2)...")
        book_vectors = [self.get_book_hofstede(b) for b in books]
        
        user_ratings_group = df_train.groupby('user_idx')
        user_profiles = {}
        
        for u, group in user_ratings_group:
            high_rated = group[group['rating'] >= 3.0]
            if len(high_rated) == 0:
                high_rated = group
                
            vectors = []
            for _, row in high_rated.iterrows():
                b_idx = int(row['book_idx'])
                vectors.append(book_vectors[b_idx])
            
            if vectors:
                user_profiles[u] = np.mean(vectors, axis=0)
            else:
                user_profiles[u] = self.global_average_hofstede.copy()
                
        return user_profiles, book_vectors

    def fit(self, df_train, books, book_id_to_idx, user_profiles, book_vectors):
        """Train CulturallyAwareFMv2 with explicit distance features via SGD."""
        self.num_users = int(df_train['user_idx'].max()) + 1
        self.num_books = len(books)
        
        # Features: [ User IDs (num_users) | Book IDs (num_books) | Continuous Alignment Features (20) ]
        self.d = self.num_users + self.num_books + self.num_continuous
        self.hofstede_offset = self.num_users + self.num_books
        
        print(f"Initializing FM v2 weights (total features dim = {self.d})...")
        np.random.seed(42)
        self.w0 = df_train['rating'].mean()
        self.w = np.zeros(self.d)
        self.V = np.random.normal(0.0, 0.05, (self.d, self.k))
        
        print(f"Training Culturally Aware FM v2 for {self.epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            losses = []
            
            shuffled_indices = np.random.permutation(len(df_train))
            for idx in shuffled_indices:
                row = df_train.iloc[idx]
                u_idx = int(row['user_idx'])
                b_idx = int(row['book_idx'])
                rating = float(row['rating'])
                
                u_hof = user_profiles.get(u_idx, self.global_average_hofstede)
                b_hof = book_vectors[b_idx]
                
                # Compute 20 explicit cultural alignment features
                cultural_feats = compute_cultural_alignment_features(u_hof, b_hof)
                
                # Retrieve active indices and values
                active_indices = [u_idx, self.num_users + b_idx]
                active_values = [1.0, 1.0]
                
                active_indices.extend(range(self.hofstede_offset, self.hofstede_offset + self.num_continuous))
                active_values.extend(cultural_feats)
                
                # 1. Forward Pass
                y_hat = self.w0 + sum(self.w[i] * val for i, val in zip(active_indices, active_values))
                
                sum_vx = np.zeros(self.k)
                sum_v2x2 = np.zeros(self.k)
                
                for f in range(self.k):
                    s_vx = 0.0
                    s_v2x2 = 0.0
                    for i, val in zip(active_indices, active_values):
                        term = self.V[i, f] * val
                        s_vx += term
                        s_v2x2 += term * term
                    sum_vx[f] = s_vx
                    sum_v2x2[f] = s_v2x2
                    
                interaction_sum = 0.5 * np.sum(sum_vx**2 - sum_v2x2)
                y_hat += interaction_sum
                
                y_hat = min(5.0, max(1.0, y_hat))
                
                # 2. Backpropagation / SGD Update
                error = rating - y_hat
                losses.append(error * error)
                
                self.w0 += self.lr * error
                
                for i, val in zip(active_indices, active_values):
                    self.w[i] += self.lr * (error * val - self.reg * self.w[i])
                    for f in range(self.k):
                        grad_v = error * val * (sum_vx[f] - self.V[i, f] * val)
                        self.V[i, f] += self.lr * (grad_v - self.reg * self.V[i, f])
                        
            rmse = np.sqrt(np.mean(losses))
            print(f"   Epoch {epoch}/{self.epochs} - Loss (RMSE): {rmse:.4f} in {time.time() - epoch_start:.2f}s")
            
        print(f"FM v2 Training completed in {time.time() - start_time:.2f} seconds.")

    def predict(self, user_id, book_idx, user_hof, book_hof):
        """Predict rating for user-book pair, supporting cold start."""
        active_indices = []
        active_values = []
        
        # User ID (only for known integer training indices)
        if user_id is not None and isinstance(user_id, (int, np.integer)) and 0 <= user_id < self.num_users:
            active_indices.append(int(user_id))
            active_values.append(1.0)
            
        active_indices.append(self.num_users + int(book_idx))
        active_values.append(1.0)
        
        # 20 continuous alignment features
        cultural_feats = compute_cultural_alignment_features(user_hof, book_hof)
        active_indices.extend(range(self.hofstede_offset, self.hofstede_offset + self.num_continuous))
        active_values.extend(cultural_feats)
        
        y_hat = self.w0 + sum(self.w[i] * val for i, val in zip(active_indices, active_values))
        
        sum_vx = np.zeros(self.k)
        sum_v2x2 = np.zeros(self.k)
        
        for f in range(self.k):
            s_vx = 0.0
            s_v2x2 = 0.0
            for i, val in zip(active_indices, active_values):
                term = self.V[i, f] * val
                s_vx += term
                s_v2x2 += term * term
            sum_vx[f] = s_vx
            sum_v2x2[f] = s_v2x2
            
        interaction_sum = 0.5 * np.sum(sum_vx**2 - sum_v2x2)
        y_hat += interaction_sum
        
        return min(5.0, max(1.0, y_hat))
