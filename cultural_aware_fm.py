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

class CulturallyAwareFM:
    def __init__(self, num_factors=10, learning_rate=0.005, regularization=0.05, epochs=10):
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
        self.d = 0 # total features dimension
        
    def load_hofstede_csv(self, file_path):
        """Load and clean Hofstede CSV data, imputing any missing values with column medians."""
        print("Loading Hofstede scores from CSV...")
        if not os.path.exists(file_path):
            print(f"Warning: Hofstede CSV not found at {file_path}. Using default global averages.")
            return
            
        df = pd.read_csv(file_path)
        
        # Clean column names
        df.columns = [c.strip().lower() for c in df.columns]
        
        dimensions = ['pdi', 'idv', 'mas', 'uai', 'lto', 'ivr']
        
        # Convert dimensions to float, coercing errors to NaN
        for dim in dimensions:
            df[dim] = pd.to_numeric(df[dim], errors='coerce')
            
        # Impute missing values with column medians
        medians = df[dimensions].median()
        df[dimensions] = df[dimensions].fillna(medians)
        
        # Build map
        self.hofstede_map = {}
        for _, row in df.iterrows():
            country = str(row['country']).strip().lower()
            vector = np.array([row[d] for d in dimensions], dtype=float)
            self.hofstede_map[country] = vector
            
        # Compute global average profile
        if self.hofstede_map:
            self.global_average_hofstede = np.mean(list(self.hofstede_map.values()), axis=0)
        print(f"Loaded Hofstede profiles for {len(self.hofstede_map)} countries. Global Average: {self.global_average_hofstede}")

    def get_book_hofstede(self, book):
        """Retrieve the Hofstede vector of a book based on its country or language code."""
        # 1. Try mapping the book country code (ISO-2)
        cc = str(book.get("country_code", "")).strip().lower()
        country_name = ISO2_TO_COUNTRY.get(cc)
        if country_name and country_name in self.hofstede_map:
            return self.hofstede_map[country_name]
            
        # 2. Try raw country string search
        if cc in self.hofstede_map:
            return self.hofstede_map[cc]
            
        # 3. Try fallback using language code mapping
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
        print("Propagating book metadata to build user cultural profiles...")
        # Precompute book vectors
        book_vectors = [self.get_book_hofstede(b) for b in books]
        
        user_ratings_group = df_train.groupby('user_idx')
        user_profiles = {}
        
        for u, group in user_ratings_group:
            # Look at highly rated books first (rating >= 3.0)
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
        """Train the Culturally Aware Factorization Machine via Stochastic Gradient Descent."""
        self.num_users = int(df_train['user_idx'].max()) + 1
        self.num_books = len(books)
        
        # Feature vector offsets
        # Indices: [ 0..num_users-1 (User IDs) | num_users..num_users+num_books-1 (Book IDs) | offset..offset+5 (User Hofstede) | offset+6..offset+11 (Book Hofstede) ]
        self.d = self.num_users + self.num_books + 12
        self.hofstede_offset = self.num_users + self.num_books
        
        print(f"Initializing Factorization Machine weights (features dim = {self.d})...")
        np.random.seed(42)
        self.w0 = df_train['rating'].mean()
        self.w = np.zeros(self.d)
        self.V = np.random.normal(0.0, 0.05, (self.d, self.k))
        
        print(f"Training Culturally Aware FM for {self.epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            losses = []
            
            # Shuffle ratings for SGD training stability
            shuffled_indices = np.random.permutation(len(df_train))
            for idx in shuffled_indices:
                row = df_train.iloc[idx]
                u_idx = int(row['user_idx'])
                b_idx = int(row['book_idx'])
                rating = float(row['rating'])
                
                u_hof = user_profiles.get(u_idx, self.global_average_hofstede)
                b_hof = book_vectors[b_idx]
                
                # Retrieve sparse feature representation
                active_indices = [u_idx, self.num_users + b_idx]
                active_values = [1.0, 1.0]
                
                # Append continuous Hofstede features (normalized to 0-1 scale for stability)
                active_indices.extend(range(self.hofstede_offset, self.hofstede_offset + 12))
                active_values.extend(list(u_hof / 100.0) + list(b_hof / 100.0))
                
                # 1. Forward Pass
                # Linear part
                y_hat = self.w0 + sum(self.w[idx] * val for idx, val in zip(active_indices, active_values))
                
                # Pairwise interactions: sum_{f} 0.5 * ( (sum V_{i,f} x_i)^2 - sum V_{i,f}^2 x_i^2 )
                sum_vx = np.zeros(self.k)
                sum_v2x2 = np.zeros(self.k)
                
                for f in range(self.k):
                    s_vx = 0.0
                    s_v2x2 = 0.0
                    for idx, val in zip(active_indices, active_values):
                        term = self.V[idx, f] * val
                        s_vx += term
                        s_v2x2 += term * term
                    sum_vx[f] = s_vx
                    sum_v2x2[f] = s_v2x2
                    
                interaction_sum = 0.5 * np.sum(sum_vx**2 - sum_v2x2)
                y_hat += interaction_sum
                
                # Clip prediction to scale [1.0, 5.0]
                y_hat = min(5.0, max(1.0, y_hat))
                
                # 2. Backpropagation / SGD Update
                error = rating - y_hat
                losses.append(error * error)
                
                # Update global bias
                self.w0 += self.lr * error
                
                # Update linear weights and latent factors
                for idx, val in zip(active_indices, active_values):
                    # Linear weight update
                    self.w[idx] += self.lr * (error * val - self.reg * self.w[idx])
                    
                    # Latent factor update
                    for f in range(self.k):
                        grad_v = error * val * (sum_vx[f] - self.V[idx, f] * val)
                        self.V[idx, f] += self.lr * (grad_v - self.reg * self.V[idx, f])
                        
            rmse = np.sqrt(np.mean(losses))
            print(f"   Epoch {epoch}/{self.epochs} - Loss (RMSE): {rmse:.4f} in {time.time() - epoch_start:.2f}s")
            
        print(f"FM Training completed in {time.time() - start_time:.2f} seconds.")

    def predict(self, user_id, book_idx, user_hof, book_hof):
        """
        Predict explicit rating for user-book pair.
        Supports cold start (user_id is None or unknown):
        if user_id is None, the user ID one-hot index is omitted.
        """
        # Linear part
        active_indices = []
        active_values = []
        
        # User ID (only for known integer training indices)
        if user_id is not None and isinstance(user_id, (int, np.integer)) and 0 <= user_id < self.num_users:
            active_indices.append(int(user_id))
            active_values.append(1.0)
            
        # Book index
        active_indices.append(self.num_users + int(book_idx))
        active_values.append(1.0)
        
        # Hofstede continuous vectors
        active_indices.extend(range(self.hofstede_offset, self.hofstede_offset + 12))
        active_values.extend(list(user_hof / 100.0) + list(book_hof / 100.0))
        
        y_hat = self.w0 + sum(self.w[idx] * val for idx, val in zip(active_indices, active_values))
        
        # Pairwise interaction part
        sum_vx = np.zeros(self.k)
        sum_v2x2 = np.zeros(self.k)
        
        for f in range(self.k):
            s_vx = 0.0
            s_v2x2 = 0.0
            for idx, val in zip(active_indices, active_values):
                term = self.V[idx, f] * val
                s_vx += term
                s_v2x2 += term * term
            sum_vx[f] = s_vx
            sum_v2x2[f] = s_v2x2
            
        interaction_sum = 0.5 * np.sum(sum_vx**2 - sum_v2x2)
        y_hat += interaction_sum
        
        return min(5.0, max(1.0, y_hat))
