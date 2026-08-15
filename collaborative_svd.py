import numpy as np
import pandas as pd
from surprise import SVDpp, Reader, Dataset

class SurpriseSVDpp:
    def __init__(self, n_factors=15, n_epochs=10, lr_all=0.007, reg_all=0.02, random_state=42):
        """
        Wrapper around scikit-surprise SVD++ implementation.
        - n_factors: Dimensionality of latent factor spaces.
        - n_epochs: Number of SGD epochs to run.
        - lr_all: Learning rate for parameters.
        - reg_all: Regularization term.
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state
        self.model = None
        self.global_mean = 3.5
        
    def fit(self, df_train):
        """
        Fit the SVD++ model on rating interactions.
        df_train: pandas.DataFrame containing ['user_id', 'book_id', 'rating'] columns.
        """
        if len(df_train) == 0:
            print("Warning: Empty training dataset for SVD++.")
            return
            
        self.global_mean = df_train['rating'].mean()
        
        # Surprise requires ID columns to be strings to prevent mapping errors
        df_surprise = pd.DataFrame({
            'user_id': df_train['user_id'].astype(str),
            'book_id': df_train['book_id'].astype(str),
            'rating': df_train['rating'].astype(float)
        })
        
        # Build dataset
        reader = Reader(rating_scale=(1.0, 5.0))
        data = Dataset.load_from_df(df_surprise[['user_id', 'book_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        
        # Fit SVDpp
        self.model = SVDpp(
            n_factors=self.n_factors, 
            n_epochs=self.n_epochs, 
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state
        )
        
        print("Training Surprise SVD++ (Collaborative Latent SVD++)...")
        self.model.fit(trainset)
        print("SVD++ training completed.")
        
    def predict(self, user_id, book_id):
        """
        Predict explicit rating for user_id and book_id.
        """
        if self.model is None:
            return self.global_mean
            
        # Predict using string keys matching training set conversion
        prediction = self.model.predict(str(user_id), str(book_id))
        return prediction.est
