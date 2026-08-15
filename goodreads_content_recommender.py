import json
import os
import re
import sys
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Try to import PyTorch, Sentence Transformers, and FAISS for GPU acceleration
try:
    import torch
    from sentence_transformers import SentenceTransformer
    import faiss
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False

# List of non-informative shelves/tags
NOISY_TAGS = {
    'to-read', 'currently-reading', 'owned', 'favorites', 'books-i-own', 
    'my-library', 'library', 'all-time-favorites', 'wish-list', 'to-buy', 
    're-read', 'audiobook', 'audiobooks', 'audio', 'ebook', 'ebooks', 
    'kindle', 'paperback', 'hardcover', 'default', 'own-it', 'on-hold',
    'my-books', 'i-own', 'favourites', 'read-in-2012', 'read-in-2013',
    'read-in-2014', 'read-in-2015', 'read-in-2016', 'read-in-2017',
    'read-in-2018', 'read-in-2019', 'read-in-2020', 'read-in-2021',
    'read-in-2022', 'read-in-2023', 'read-in-2024', 'read-in-2025',
    'read-in-2026', 'maybe', 'abandoned', 'did-not-finish', 'dnf',
    'tbr', 'shelfari-favorites', 'shelfari-wishlist', 'owned-books',
    'e-book', 'e-books', 'historical', 'fiction', 'novel', 'novels'
}

def clean_html(text):
    """Remove HTML tags from text."""
    if not text:
        return ""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def load_authors(file_path):
    """Load author lookup map."""
    print("Loading authors map from goodreads_book_authors.json...")
    start_time = time.time()
    authors_map = {}
    if not os.path.exists(file_path):
        print(f"Warning: Authors file not found at {file_path}")
        return authors_map
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                author_id = data.get("author_id")
                name = data.get("name")
                if author_id and name:
                    authors_map[author_id] = name
            except Exception:
                continue
    print(f"Loaded {len(authors_map)} authors in {time.time() - start_time:.2f} seconds.")
    return authors_map

def load_genres(file_path):
    """Load genre lookup map."""
    print("Loading genres map from goodreads_book_genres_initial.json...")
    start_time = time.time()
    genres_map = {}
    if not os.path.exists(file_path):
        print(f"Warning: Genres file not found at {file_path}")
        return genres_map
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                book_id = data.get("book_id")
                genres_dict = data.get("genres", {})
                genres_list = []
                for genre_str, count in genres_dict.items():
                    if count > 0:
                        sub_genres = [g.strip().lower() for g in genre_str.split(",")]
                        genres_list.extend(sub_genres)
                if book_id and genres_list:
                    genres_map[book_id] = list(set(genres_list))
            except Exception:
                continue
    print(f"Loaded genres for {len(genres_map)} books in {time.time() - start_time:.2f} seconds.")
    return genres_map

def load_books_dataset(file_path, authors_map, genres_map, limit=40000):
    """Stream book JSON lines and parse metadata."""
    print(f"Streaming books from {file_path} (limit={limit})...")
    start_time = time.time()
    books = []
    
    if not os.path.exists(file_path):
        print(f"Error: Books file not found at {file_path}")
        sys.exit(1)
        
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(books) >= limit:
                break
            try:
                data = json.loads(line)
                book_id = data.get("book_id")
                title = data.get("title")
                description = data.get("description", "")
                country_code = data.get("country_code", "US")
                language_code = data.get("language_code", "eng")
                
                # Require title, valid book ID, and description
                if not title or not book_id or len(description.strip()) < 20:
                    continue
                
                # Map authors names
                author_names = []
                for author_info in data.get("authors", []):
                    a_id = author_info.get("author_id")
                    if a_id in authors_map:
                        author_names.append(authors_map[a_id])
                
                # Retrieve genres
                genres = genres_map.get(book_id, [])
                
                # Filter shelves
                tags = []
                for shelf in data.get("popular_shelves", []):
                    name = shelf.get("name", "").lower().strip()
                    try:
                        count = int(shelf.get("count", 0))
                    except ValueError:
                        count = 0
                    if name and count > 1 and name not in NOISY_TAGS and len(name) > 2:
                        tags.append(name)
                
                books.append({
                    "book_id": book_id,
                    "title": title,
                    "authors": author_names,
                    "genres": genres,
                    "tags": tags,
                    "description": clean_html(description),
                    "country_code": country_code,
                    "language_code": language_code,
                    "average_rating": data.get("average_rating", "0.0"),
                    "ratings_count": data.get("ratings_count", "0")
                })
            except Exception:
                continue
                
    print(f"Loaded {len(books)} books in {time.time() - start_time:.2f} seconds.")
    return books

def build_content_soup(book):
    """Create a weighted TF-IDF content soup."""
    title_part = (book["title"] + " ") * 2
    author_tokens = [a.lower().replace(" ", "").replace(".", "") for a in book["authors"]]
    author_part = (" ".join(author_tokens) + " ") * 3
    genre_tokens = [g.replace(" ", "-") for g in book["genres"]]
    genre_part = (" ".join(genre_tokens) + " ") * 2
    tag_tokens = [t.replace(" ", "-") for t in book["tags"]]
    tag_part = (" ".join(tag_tokens) + " ") * 2
    
    return title_part + author_part + genre_part + tag_part + book["description"]

def load_semantic_model():
    """Load SentenceTransformer on M3 GPU (MPS) or fallback to CPU."""
    if not HAS_SEMANTIC:
        print("Sentence-Transformers or FAISS not installed. Using TF-IDF mode.")
        return None
    
    print("Loading SentenceTransformer model ('all-MiniLM-L6-v2')...")
    start = time.time()
    # Target CPU for stability on macOS, avoiding MPS driver segmentation faults
    device = "cpu"
    print(f"Targeting device: {device}")
    
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    print(f"Model loaded in {time.time() - start:.2f} seconds.")
    return model

def generate_semantic_embeddings(books, model):
    """Encode descriptions and build FAISS index using normalized L2 cosine search."""
    print(f"Encoding {len(books)} books with SentenceTransformer...")
    start = time.time()
    
    texts = []
    for b in books:
        authors_str = ", ".join(b["authors"]) if b["authors"] else "Unknown Author"
        genres_str = ", ".join(b["genres"]) if b["genres"] else "No genres listed"
        texts.append(f"{b['title']} by {authors_str}. Genres: {genres_str}. Description: {b['description']}")
        
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    
    # Normalize for cosine similarity lookups in IndexFlatIP
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    print(f"FAISS semantic index built in {time.time() - start:.2f} seconds.")
    return embeddings, index

def fit_tfidf_recommender(books):
    """Fit TF-IDF matrix as fallback content representation."""
    print("Fitting TF-IDF text representation...")
    start_time = time.time()
    
    soups = [build_content_soup(b) for b in books]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=25000)
    tfidf_matrix = vectorizer.fit_transform(soups)
    
    print(f"TF-IDF matrix built with shape {tfidf_matrix.shape} in {time.time() - start_time:.2f} seconds.")
    return vectorizer, tfidf_matrix

def predict_rating(user_history, book_idx, model_context):
    """
    Predict explicit rating for book_idx based on user's rating history.
    model_context is a dictionary containing fitted models and embeddings.
    """
    if not user_history:
        return 3.5  # default baseline
        
    similarities = []
    ratings = []
    
    is_semantic = model_context.get("is_semantic", False)
    
    if is_semantic:
        embeddings = model_context["embeddings"]
        target_emb = embeddings[book_idx].reshape(1, -1)
        for hist_idx, r in user_history:
            hist_emb = embeddings[hist_idx].reshape(1, -1)
            # Dot product computes cosine similarity for normalized L2 vectors
            sim = float(np.dot(target_emb, hist_emb.T)[0, 0])
            sim = max(0.0, sim)
            similarities.append(sim)
            ratings.append(r)
    else:
        tfidf_matrix = model_context["tfidf_matrix"]
        target_vec = tfidf_matrix[book_idx]
        for hist_idx, r in user_history:
            sim = float(target_vec.dot(tfidf_matrix[hist_idx].T).toarray()[0, 0])
            sim = max(0.0, sim)
            similarities.append(sim)
            ratings.append(r)
            
    sum_sims = sum(similarities)
    if sum_sims > 1e-5:
        return sum(sim * r for sim, r in zip(similarities, ratings)) / sum_sims
    else:
        return sum(ratings) / len(ratings)

def recommend_books(query_title, books, model_context, k=5):
    """Query recommendations using either FAISS semantic search or TF-IDF matrix dot products."""
    query_title_lower = query_title.lower().strip()
    matches = []
    for idx, b in enumerate(books):
        if query_title_lower in b["title"].lower():
            matches.append((idx, b))
            
    if not matches:
        return None, None
        
    if len(matches) > 1:
        print(f"\n🔍 Found multiple matches for '{query_title}':")
        for i, (idx, b) in enumerate(matches[:5]):
            authors_str = ", ".join(b["authors"]) if b["authors"] else "Unknown Author"
            print(f"  [{i+1}] {b['title']} by {authors_str}")
        print(f"Selected match: '{matches[0][1]['title']}'")
        
    query_idx, query_book = matches[0]
    
    is_semantic = model_context.get("is_semantic", False)
    
    if is_semantic:
        embeddings = model_context["embeddings"]
        index = model_context["index"]
        query_vector = embeddings[query_idx].reshape(1, -1)
        
        # FAISS search
        # D is cosine similarity distances, I is item indices
        D, I = index.search(query_vector, k + 1)
        
        recommendations = []
        for i in range(len(I[0])):
            idx = int(I[0][i])
            if idx == query_idx:
                continue
            if len(recommendations) >= k:
                break
            recommendations.append({
                "book": books[idx],
                "similarity": float(D[0][i])
            })
    else:
        tfidf_matrix = model_context["tfidf_matrix"]
        query_vector = tfidf_matrix[query_idx]
        similarities = tfidf_matrix.dot(query_vector.T).toarray().ravel()
        
        related_indices = np.argsort(similarities)[::-1]
        recommendations = []
        for idx in related_indices:
            if idx == query_idx:
                continue
            if len(recommendations) >= k:
                break
            recommendations.append({
                "book": books[idx],
                "similarity": float(similarities[idx])
            })
            
    return query_book, recommendations

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    goodreads_dir = os.path.join(base_dir, "goodreads")
    
    authors_path = os.path.join(goodreads_dir, "goodreads_book_authors.json")
    genres_path = os.path.join(goodreads_dir, "goodreads_book_genres_initial.json")
    books_path = os.path.join(goodreads_dir, "goodreads_books.json")
    
    authors_map = load_authors(authors_path)
    genres_map = load_genres(genres_path)
    
    limit = 20000
    books = load_books_dataset(books_path, authors_map, genres_map, limit=limit)
    
    if not books:
        print("Error: No books loaded.")
        return
        
    model_context = {"is_semantic": False}
    semantic_model = load_semantic_model()
    
    if semantic_model:
        embeddings, index = generate_semantic_embeddings(books, semantic_model)
        model_context.update({
            "is_semantic": True,
            "embeddings": embeddings,
            "index": index
        })
    else:
        vectorizer, tfidf_matrix = fit_tfidf_recommender(books)
        model_context.update({
            "tfidf_matrix": tfidf_matrix,
            "vectorizer": vectorizer
        })
        
    print("\nContent Recommender system ready! Enter 'exit' or 'quit' to end.")
    while True:
        try:
            query = input("\nEnter a book title: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                break
                
            query_book, recommendations = recommend_books(query, books, model_context, k=5)
            if query_book and recommendations:
                print(f"\n📖 TARGET BOOK: {query_book['title']}")
                print(f"   Genres: {', '.join(query_book['genres'])}")
                print(f"   Description: {query_book['description'][:150]}...\n")
                print("📚 RECOMMENDATIONS:")
                for i, rec in enumerate(recommendations, 1):
                    b = rec["book"]
                    print(f"   {i}. {b['title']} (Score: {rec['similarity']*100:.1f}%)")
                    print(f"      Genres: {', '.join(b['genres'])}")
                    print(f"      Description: {b['description'][:120]}...\n")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
