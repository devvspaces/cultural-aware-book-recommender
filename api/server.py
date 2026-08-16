"""
FastAPI Backend Server for Culturally Aware Literary Recommender
================================================================
Endpoints:
- GET  /api/health            -> System health & catalog stats
- GET  /api/countries         -> List of 119 Hofstede countries
- POST /api/onboard           -> Initialize user session with country & cold-start recommendations
- POST /api/rate              -> Submit a rating, update user cultural profile, return refreshed recommendations
- GET  /api/recommend         -> Retrieve Top-K hybrid recommendations
- GET  /api/search            -> Search book catalog by title, author, or genre
- GET  /api/profile/{user_id} -> User's rating history & current Hofstede profile
- GET  /api/cover/{book_id}   -> Fetch cover URL with caching
"""

import os
import sys
import time
import json
import uuid
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from surprise import dump

# Base paths
API_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(API_DIR) # project directory
BASE_DIR = os.path.dirname(PROJECT_DIR) # repository root directory
GOODREADS_DIR = os.path.join(BASE_DIR, "goodreads")
sys.path.append(PROJECT_DIR)

import goodreads_content_recommender as gcr
from hybrid_recommender import HybridRecommender

app = FastAPI(
    title="African & Global Culturally Aware Recommender API",
    description="Backend API powering the Culturally Aware Hybrid Recommendation Platform",
    version="1.0.0"
)

# Enable CORS for React frontend (Vite dev server and production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
DATA_STATE: Dict[str, Any] = {
    "hybrid_model": None,
    "books": [],
    "book_id_to_idx": {},
    "book_csv_to_idx": {},
    "hofstede_map": {},
    "country_list": [],
    "sessions": {}, # {user_id: {"country": str, "user_hof": np.ndarray, "history": [(book_idx, rating)]}}
    "cover_cache": {},
}

DIMENSIONS = ['pdi', 'idv', 'mas', 'uai', 'lto', 'ivr']
DIMENSION_LABELS = {
    'pdi': 'Power Distance (Hierarchy)',
    'idv': 'Individualism vs Collectivism',
    'mas': 'Achievement & Competition',
    'uai': 'Uncertainty Avoidance (Structure)',
    'lto': 'Long-Term Orientation (Tradition vs Future)',
    'ivr': 'Indulgence vs Restraint'
}

# ─── Pydantic Models ─────────────────────────────────────────────────────────
class OnboardRequest(BaseModel):
    country: str
    language: Optional[str] = "eng"

class RatingRequest(BaseModel):
    user_id: str
    book_id: str
    rating: float

# ─── Startup Event ───────────────────────────────────────────────────────────
@app.on_event("startup")
def load_data_and_models():
    print("\n" + "="*80)
    print("  INITIALIZING RECOMMENDER BACKEND SERVER...")
    print("="*80)
    
    authors_path = os.path.join(GOODREADS_DIR, "goodreads_book_authors.json")
    genres_path = os.path.join(GOODREADS_DIR, "goodreads_book_genres_initial.json")
    books_path = os.path.join(GOODREADS_DIR, "goodreads_books.json")
    book_id_map_path = os.path.join(GOODREADS_DIR, "book_id_map.csv")
    interactions_path = os.path.join(GOODREADS_DIR, "goodreads_interactions.csv")
    hofstede_path = os.path.join(PROJECT_DIR, "hofstede.csv")
    
    # 1. Load Hofstede
    df_hof = pd.read_csv(hofstede_path)
    df_hof.columns = [c.strip().lower() for c in df_hof.columns]
    for d in DIMENSIONS:
        df_hof[d] = pd.to_numeric(df_hof[d], errors='coerce').fillna(df_hof[d].median())
        
    hofstede_map = {}
    country_list = []
    for _, row in df_hof.iterrows():
        c_name = str(row['country']).strip()
        v = row[DIMENSIONS].values.astype(float)
        hofstede_map[c_name.lower()] = v
        country_list.append({
            "name": c_name,
            "code": c_name.lower(),
            "pdi": float(v[0]), "idv": float(v[1]), "mas": float(v[2]),
            "uai": float(v[3]), "lto": float(v[4]), "ivr": float(v[5])
        })
    country_list.sort(key=lambda x: x['name'])
    DATA_STATE["hofstede_map"] = hofstede_map
    DATA_STATE["country_list"] = country_list
    print(f"Loaded {len(country_list)} Hofstede country profiles.")
    
    # 2. Check for Pre-Serialized Artifacts (Fast Cloud Mode)
    artifacts_catalog = os.path.join(PROJECT_DIR, "artifacts", "books_catalog.json")
    artifacts_fm_weights = os.path.join(PROJECT_DIR, "artifacts", "fm_v2_weights.npz")
    artifacts_svd = os.path.join(PROJECT_DIR, "artifacts", "svd_model.pkl")
    
    if os.path.exists(artifacts_catalog) and os.path.exists(artifacts_fm_weights):
        print("\n⚡ FAST CLOUD MODE: Loading pre-trained models and catalog from artifacts/...")
        with open(artifacts_catalog, 'r', encoding='utf-8') as f:
            books = json.load(f)
            
        book_id_to_idx = {b['book_id']: idx for idx, b in enumerate(books)}
        DATA_STATE["books"] = books
        DATA_STATE["book_id_to_idx"] = book_id_to_idx
        print(f"Loaded {len(books):,} preprocessed books.")
        
        # Initialize and restore Hybrid Recommender
        hybrid = HybridRecommender(alpha=0.80, threshold_t=1, fm_factors=10, svd_factors=10)
        hybrid.load_hofstede_csv(hofstede_path)
        hybrid.books = books
        hybrid.book_id_to_idx = book_id_to_idx
        
        # Load weights
        npz = np.load(artifacts_fm_weights)
        hybrid.fm_v2.w0 = float(npz['w0'])
        hybrid.fm_v2.w = npz['w']
        hybrid.fm_v2.V = npz['V']
        hybrid.fm_v2.global_average_hofstede = npz['global_average_hofstede']
        
        config_path = os.path.join(PROJECT_DIR, "artifacts", "recommender_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            hybrid.fm_v2.num_users = cfg.get("num_users", 5951)
            hybrid.fm_v2.num_books = cfg.get("num_books", len(books))
            hybrid.fm_v2.num_continuous = cfg.get("num_continuous", 20)
        else:
            hybrid.fm_v2.num_users = len(hybrid.fm_v2.w) - len(books) - 20
            hybrid.fm_v2.num_books = len(books)
            hybrid.fm_v2.num_continuous = 20
            
        hybrid.fm_v2.hofstede_offset = hybrid.fm_v2.num_users + hybrid.fm_v2.num_books
        hybrid.fm_v2.total_features = hybrid.fm_v2.hofstede_offset + hybrid.fm_v2.num_continuous
        hybrid.fm_v2.is_fitted = True
        
        # Load SVD++
        if os.path.exists(artifacts_svd):
            _, loaded_svd = dump.load(artifacts_svd)
            hybrid.svd_model.model = loaded_svd
        
        # Compute book cultural vectors
        hybrid.book_vectors = [
            hybrid.fm_v2.get_book_hofstede(b) for b in books
        ]
        hybrid.is_fitted = True
        DATA_STATE["hybrid_model"] = hybrid
        print("✓ Pre-trained Hybrid Recommender restored in <2 seconds!")
        return
        
    # 3. Fallback to Raw Data Parsing (if raw files exist)
    if os.path.exists(book_id_map_path) and os.path.exists(books_path):
        df_map = pd.read_csv(book_id_map_path)
        df_map['book_id'] = df_map['book_id'].astype(str)
        book_id_str_to_csv = dict(zip(df_map['book_id'], df_map['book_id_csv']))
        
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
        book_id_to_idx = {b['book_id']: idx for idx, b in enumerate(books)}
        book_csv_to_idx = {b['book_id_csv']: idx for idx, b in enumerate(books)}
        DATA_STATE["books"] = books
        DATA_STATE["book_id_to_idx"] = book_id_to_idx
        DATA_STATE["book_csv_to_idx"] = book_csv_to_idx
        
        ratings = []
        for chunk in pd.read_csv(interactions_path, chunksize=100000):
            filtered = chunk[(chunk['book_id'].isin(book_csv_ids_set)) & (chunk['rating'] > 0)]
            for _, row in filtered.iterrows():
                ratings.append({
                    'user_id': int(row['user_id']),
                    'book_id': int(row['book_id']),
                    'rating': float(row['rating'])
                })
                if len(ratings) >= 50000:
                    break
            if len(ratings) >= 50000:
                break
                
        df_ratings = pd.DataFrame(ratings)
        unique_users = df_ratings['user_id'].unique()
        user_to_idx = {orig_u: idx for idx, orig_u in enumerate(unique_users)}
        df_ratings['user_idx'] = df_ratings['user_id'].map(user_to_idx)
        df_ratings['book_idx'] = df_ratings['book_id'].map(book_csv_to_idx)
        
        hybrid = HybridRecommender(alpha=0.80, threshold_t=1, fm_factors=10, svd_factors=10, epochs=6)
        hybrid.load_hofstede_csv(hofstede_path)
        hybrid.fit(df_ratings, books, book_csv_to_idx)
        DATA_STATE["hybrid_model"] = hybrid
        print("\n✓ FASTAPI SERVER INITIALIZATION COMPLETE — READY FOR TRAFFIC!")

# ─── Helpers ─────────────────────────────────────────────────────────────────
def format_cultural_profile_dict(vector: np.ndarray) -> Dict[str, Any]:
    return {
        "pdi": round(float(vector[0]), 1),
        "idv": round(float(vector[1]), 1),
        "mas": round(float(vector[2]), 1),
        "uai": round(float(vector[3]), 1),
        "lto": round(float(vector[4]), 1),
        "ivr": round(float(vector[5]), 1),
        "labels": DIMENSION_LABELS
    }

# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/api/health")
def get_health():
    return {
        "status": "online",
        "total_books": len(DATA_STATE["books"]),
        "total_countries": len(DATA_STATE["country_list"]),
        "active_sessions": len(DATA_STATE["sessions"]),
        "model": "Culturally Aware Hybrid Recommender (FM v2 + SVD++)"
    }

@app.get("/api/countries")
def get_countries():
    return {"countries": DATA_STATE["country_list"]}

@app.post("/api/onboard")
def onboard_user(req: OnboardRequest):
    c_key = req.country.strip().lower()
    hof_map = DATA_STATE["hofstede_map"]
    hybrid: HybridRecommender = DATA_STATE["hybrid_model"]
    
    # Lookup country vector
    if c_key in hof_map:
        user_hof = hof_map[c_key].copy()
    else:
        user_hof = hybrid.fm_v2.global_average_hofstede.copy()
        
    user_id = f"user_{uuid.uuid4().hex[:10]}"
    
    # Store session
    DATA_STATE["sessions"][user_id] = {
        "country": req.country,
        "language": req.language or "eng",
        "user_hof": user_hof,
        "history": [] # [(book_idx, rating)]
    }
    
    # Generate Cold-Start Top 12
    recs = hybrid.recommend_top_k(user_id=None, user_hof=user_hof, k=12)
    
    return {
        "user_id": user_id,
        "country": req.country,
        "cultural_profile": format_cultural_profile_dict(user_hof),
        "recommendations": recs
    }

@app.post("/api/rate")
def rate_book(req: RatingRequest):
    user_id = req.user_id
    sessions = DATA_STATE["sessions"]
    book_id_to_idx = DATA_STATE["book_id_to_idx"]
    hybrid: HybridRecommender = DATA_STATE["hybrid_model"]
    
    if user_id not in sessions:
        # Create session fallback
        sessions[user_id] = {
            "country": "United States",
            "language": "eng",
            "user_hof": hybrid.fm_v2.global_average_hofstede.copy(),
            "history": []
        }
        
    if req.book_id not in book_id_to_idx:
        raise HTTPException(status_code=404, detail="Book ID not found in catalog.")
        
    b_idx = book_id_to_idx[req.book_id]
    rating = min(5.0, max(1.0, float(req.rating)))
    
    # Update hybrid internal history & recompute Hofstede profile
    updated_hof = hybrid.add_user_rating(user_id, b_idx, rating)
    sessions[user_id]["user_hof"] = updated_hof
    sessions[user_id]["history"].append({
        "book_idx": b_idx,
        "book_id": req.book_id,
        "title": DATA_STATE["books"][b_idx].get("title", ""),
        "authors": DATA_STATE["books"][b_idx].get("authors_names", []),
        "rating": rating,
        "rated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Generate refreshed recommendations (Top 12)
    recs = hybrid.recommend_top_k(user_id=user_id, user_hof=updated_hof, k=12, filter_rated=True)
    
    return {
        "status": "success",
        "user_id": user_id,
        "total_rated": len(sessions[user_id]["history"]),
        "cultural_profile": format_cultural_profile_dict(updated_hof),
        "recommendations": recs
    }

@app.get("/api/recommend")
def get_recommendations(user_id: Optional[str] = None, k: int = Query(12, ge=1, le=50)):
    hybrid: HybridRecommender = DATA_STATE["hybrid_model"]
    sessions = DATA_STATE["sessions"]
    
    user_hof = None
    if user_id and user_id in sessions:
        user_hof = sessions[user_id]["user_hof"]
    else:
        user_hof = hybrid.fm_v2.global_average_hofstede
        
    recs = hybrid.recommend_top_k(user_id=user_id, user_hof=user_hof, k=k, filter_rated=True)
    return {"recommendations": recs}

@app.get("/api/search")
def search_books(q: str = Query(..., min_length=1), user_id: Optional[str] = None, limit: int = 20):
    query = q.strip().lower()
    books = DATA_STATE["books"]
    hybrid: HybridRecommender = DATA_STATE["hybrid_model"]
    sessions = DATA_STATE["sessions"]
    
    user_hof = None
    if user_id and user_id in sessions:
        user_hof = sessions[user_id]["user_hof"]
    else:
        user_hof = hybrid.fm_v2.global_average_hofstede
        
    matches = []
    for idx, b in enumerate(books):
        title = b.get("title", "").lower()
        authors = " ".join(b.get("authors_names", [])).lower()
        genres = " ".join(b.get("genres", [])).lower()
        
        if query in title or query in authors or query in genres:
            pred_score = hybrid.predict(user_id, idx, user_hof)
            b_hof = hybrid.book_vectors[idx]
            eucl_d = np.linalg.norm(user_hof - b_hof) / (100.0 * np.sqrt(6.0))
            cultural_alignment = max(0.0, min(1.0, 1.0 - eucl_d))
            
            b_authors = b.get("authors", b.get("authors_names", ["Unknown Author"]))
            b_genres = b.get("genres", [])
            if not b_genres:
                b_genres = b.get("tags", [])[:3]
                
            desc = b.get("description", "")
            
            matches.append({
                "book_idx": idx,
                "book_id": b.get("book_id", ""),
                "title": b.get("title", "Unknown"),
                "authors": b_authors,
                "genres": b_genres[:3],
                "image_url": b.get("image_url", ""),
                "description": desc[:180] + ("..." if len(desc) > 180 else "") if desc else "No description available.",
                "average_rating": float(b.get("average_rating", 3.5)),
                "predicted_rating": round(float(pred_score), 2),
                "cultural_alignment": round(float(cultural_alignment * 100), 1)
            })
            if len(matches) >= limit:
                break
                
    matches.sort(key=lambda x: x["predicted_rating"], reverse=True)
    return {"query": q, "results": matches}

@app.get("/api/profile/{user_id}")
def get_user_profile(user_id: str):
    sessions = DATA_STATE["sessions"]
    if user_id not in sessions:
        raise HTTPException(status_code=404, detail="User session not found.")
        
    s = sessions[user_id]
    return {
        "user_id": user_id,
        "country": s["country"],
        "language": s["language"],
        "total_rated": len(s["history"]),
        "cultural_profile": format_cultural_profile_dict(s["user_hof"]),
        "history": s["history"]
    }

@app.get("/api/cover/{book_id}")
def get_book_cover(book_id: str):
    book_id_to_idx = DATA_STATE["book_id_to_idx"]
    if book_id not in book_id_to_idx:
        return {"cover_url": ""}
        
    b = DATA_STATE["books"][book_id_to_idx[book_id]]
    img = b.get("image_url", "")
    if img and "nophoto" not in img and "http" in img:
        return {"cover_url": img}
        
    isbn = b.get("isbn", "")
    if isbn:
        return {"cover_url": f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg"}
        
    return {"cover_url": f"https://covers.openlibrary.org/b/id/{book_id}-M.jpg"}

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

UI_DIST_DIR = os.path.join(PROJECT_DIR, "ui", "dist")

if os.path.exists(UI_DIST_DIR):
    assets_dir = os.path.join(UI_DIST_DIR, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
        
    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_frontend(full_path: str):
        # Allow API routes to be handled by FastAPI
        if full_path.startswith("api"):
            raise HTTPException(status_code=404, detail="API route not found")
            
        target_file = os.path.join(UI_DIST_DIR, full_path)
        if os.path.exists(target_file) and os.path.isfile(target_file):
            return FileResponse(target_file)
            
        index_file = os.path.join(UI_DIST_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
            
        return {"detail": "Frontend build not found. Run npm run build inside project/ui."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
