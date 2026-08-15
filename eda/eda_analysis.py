"""
Exploratory Data Analysis (EDA) for Goodreads + Hofstede Dataset
================================================================
Generates publication-ready charts and statistics for the final year project.
All figures are saved to project/eda/ as high-resolution PNGs.

Usage:
    ./venv-surprise/bin/python project/eda/eda_analysis.py
"""

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
    print("Warning: wordcloud not installed. Skipping word cloud generation.")

# ─── Global Style ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

PALETTE = sns.color_palette("viridis", 10)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved {path}")

# ═════════════════════════════════════════════════════════════════════════════
#  1. DATASET OVERVIEW & SCHEMA PROFILING
# ═════════════════════════════════════════════════════════════════════════════

def profile_datasets(goodreads_dir, hofstede_path):
    print("\n" + "="*80)
    print("  1. DATASET OVERVIEW & SCHEMA PROFILING")
    print("="*80)

    # --- 1a. Interactions CSV ---
    interactions_path = os.path.join(goodreads_dir, "goodreads_interactions.csv")
    print(f"\n[Interactions] {interactions_path}")
    print(f"  File size: {os.path.getsize(interactions_path) / 1e9:.2f} GB")
    df_head = pd.read_csv(interactions_path, nrows=5)
    print(f"  Columns: {list(df_head.columns)}")
    print(f"  Dtypes:\n{df_head.dtypes.to_string()}")

    # Count total lines (streaming)
    print("  Counting total rows (streaming)...")
    total_lines = 0
    with open(interactions_path, 'r') as f:
        for _ in f:
            total_lines += 1
    total_lines -= 1  # subtract header
    print(f"  Total interactions: {total_lines:,}")

    # --- 1b. Books JSON ---
    books_path = os.path.join(goodreads_dir, "goodreads_books.json")
    print(f"\n[Books] {books_path}")
    print(f"  File size: {os.path.getsize(books_path) / 1e9:.2f} GB")
    # Read first record for schema
    with open(books_path, 'r', encoding='utf-8') as f:
        first_book = json.loads(f.readline())
    print(f"  Top-level keys: {list(first_book.keys())}")
    # Count total books
    print("  Counting total books (streaming)...")
    total_books = 0
    with open(books_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_books += 1
    print(f"  Total books: {total_books:,}")

    # --- 1c. Other JSON files ---
    for name in ["goodreads_book_authors.json", "goodreads_book_genres_initial.json", "goodreads_book_series.json"]:
        fpath = os.path.join(goodreads_dir, name)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / 1e6
            line_count = 0
            with open(fpath, 'r') as f:
                for _ in f:
                    line_count += 1
            print(f"\n[{name}] {size_mb:.1f} MB, {line_count:,} records")

    # --- 1d. Mapping CSVs ---
    for name in ["book_id_map.csv", "user_id_map.csv"]:
        fpath = os.path.join(goodreads_dir, name)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, nrows=5)
            size_mb = os.path.getsize(fpath) / 1e6
            total = 0
            with open(fpath, 'r') as f:
                for _ in f:
                    total += 1
            total -= 1
            print(f"\n[{name}] {size_mb:.1f} MB, {total:,} rows")
            print(f"  Columns: {list(df.columns)}")

    # --- 1e. Hofstede CSV ---
    print(f"\n[Hofstede] {hofstede_path}")
    df_hof = pd.read_csv(hofstede_path)
    print(f"  Shape: {df_hof.shape}")
    print(f"  Columns: {list(df_hof.columns)}")
    print(f"  Missing values:\n{df_hof.isnull().sum().to_string()}")
    print(f"  Countries: {len(df_hof)}")

    return total_lines, total_books


# ═════════════════════════════════════════════════════════════════════════════
#  2. RATING DISTRIBUTION ANALYSIS (streamed from interactions)
# ═════════════════════════════════════════════════════════════════════════════

def analyze_ratings(goodreads_dir, total_interactions, total_books_count):
    print("\n" + "="*80)
    print("  2. RATING DISTRIBUTION ANALYSIS")
    print("="*80)

    interactions_path = os.path.join(goodreads_dir, "goodreads_interactions.csv")

    # Stream interactions to compute distributions without loading full 4GB
    rating_counts = Counter()  # {1: N, 2: N, ...}
    user_rating_counts = Counter()  # {user_id: count}
    book_rating_counts = Counter()  # {book_id: count}
    total_rated = 0
    unique_users = set()
    unique_books = set()

    print("  Streaming interactions for distribution analysis...")
    start = time.time()
    for chunk in pd.read_csv(interactions_path, chunksize=500000):
        rated = chunk[chunk['rating'] > 0]
        for r in rated['rating']:
            rating_counts[int(r)] += 1
        total_rated += len(rated)

        for uid in rated['user_id']:
            user_rating_counts[uid] += 1
            unique_users.add(uid)
        for bid in rated['book_id']:
            book_rating_counts[bid] += 1
            unique_books.add(bid)

    elapsed = time.time() - start
    print(f"  Processed in {elapsed:.1f}s")
    print(f"  Total rated interactions: {total_rated:,}")
    print(f"  Total interactions (incl. unrated): {total_interactions:,}")
    print(f"  Unique users with ratings: {len(unique_users):,}")
    print(f"  Unique books with ratings: {len(unique_books):,}")

    # Sparsity
    sparsity = 1.0 - (total_rated / (len(unique_users) * len(unique_books)))
    print(f"  Sparsity: {sparsity * 100:.4f}%")

    # --- 2a. Rating value histogram ---
    fig, ax = plt.subplots(figsize=(8, 5))
    stars = sorted(rating_counts.keys())
    counts = [rating_counts[s] for s in stars]
    bars = ax.bar(stars, counts, color=PALETTE[:5], edgecolor='white', linewidth=0.5)
    ax.set_xlabel("Rating (Stars)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Rating Values")
    ax.set_xticks(stars)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=9)
    save_fig(fig, "2a_rating_distribution")

    # --- 2b. Ratings per user distribution ---
    user_counts = list(user_rating_counts.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(user_counts, bins=100, color=PALETTE[2], edgecolor='white', linewidth=0.3, log=True)
    ax.set_xlabel("Number of Ratings per User")
    ax.set_ylabel("Number of Users (log scale)")
    ax.set_title("Distribution of Ratings per User")
    ax.axvline(np.median(user_counts), color='red', linestyle='--', label=f'Median: {np.median(user_counts):.0f}')
    ax.axvline(np.mean(user_counts), color='orange', linestyle='--', label=f'Mean: {np.mean(user_counts):.1f}')
    ax.legend()
    save_fig(fig, "2b_ratings_per_user")

    # --- 2c. Ratings per book distribution ---
    book_counts = list(book_rating_counts.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(book_counts, bins=100, color=PALETTE[4], edgecolor='white', linewidth=0.3, log=True)
    ax.set_xlabel("Number of Ratings per Book")
    ax.set_ylabel("Number of Books (log scale)")
    ax.set_title("Distribution of Ratings per Book")
    ax.axvline(np.median(book_counts), color='red', linestyle='--', label=f'Median: {np.median(book_counts):.0f}')
    ax.axvline(np.mean(book_counts), color='orange', linestyle='--', label=f'Mean: {np.mean(book_counts):.1f}')
    ax.legend()
    save_fig(fig, "2c_ratings_per_book")

    # Print summary statistics
    print(f"\n  Ratings per user — Mean: {np.mean(user_counts):.1f}, Median: {np.median(user_counts):.0f}, "
          f"Max: {max(user_counts)}, Std: {np.std(user_counts):.1f}")
    print(f"  Ratings per book — Mean: {np.mean(book_counts):.1f}, Median: {np.median(book_counts):.0f}, "
          f"Max: {max(book_counts)}, Std: {np.std(book_counts):.1f}")

    return user_rating_counts, book_rating_counts


# ═════════════════════════════════════════════════════════════════════════════
#  3. BOOK METADATA PROFILING (streamed from books JSON)
# ═════════════════════════════════════════════════════════════════════════════

def analyze_book_metadata(goodreads_dir):
    print("\n" + "="*80)
    print("  3. BOOK METADATA PROFILING")
    print("="*80)

    books_path = os.path.join(goodreads_dir, "goodreads_books.json")
    genres_path = os.path.join(goodreads_dir, "goodreads_book_genres_initial.json")
    authors_path = os.path.join(goodreads_dir, "goodreads_book_authors.json")

    # Stream books for metadata (sample up to 100k for efficiency)
    SAMPLE_LIMIT = 100000
    languages = Counter()
    countries = Counter()
    avg_ratings = []
    num_pages_list = []
    pub_years = []
    descriptions = []
    missing = defaultdict(int)
    total_sampled = 0

    print(f"  Streaming books metadata (up to {SAMPLE_LIMIT:,} books)...")
    start = time.time()
    with open(books_path, 'r', encoding='utf-8') as f:
        for line in f:
            if total_sampled >= SAMPLE_LIMIT:
                break
            try:
                book = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_sampled += 1

            lang = book.get("language_code", "").strip()
            if lang:
                languages[lang] += 1
            else:
                missing['language_code'] += 1

            cc = book.get("country_code", "").strip()
            if cc:
                countries[cc] += 1
            else:
                missing['country_code'] += 1

            ar = book.get("average_rating", "")
            try:
                avg_ratings.append(float(ar))
            except (ValueError, TypeError):
                missing['average_rating'] += 1

            np_val = book.get("num_pages", "")
            try:
                pages = int(np_val)
                if 1 <= pages <= 5000:
                    num_pages_list.append(pages)
            except (ValueError, TypeError):
                missing['num_pages'] += 1

            py = book.get("publication_year", "")
            try:
                year = int(py)
                if 1800 <= year <= 2026:
                    pub_years.append(year)
            except (ValueError, TypeError):
                missing['publication_year'] += 1

            desc = book.get("description", "").strip()
            if desc:
                # Clean HTML
                clean = re.sub(r'<.*?>', '', desc)
                if len(clean) > 20:
                    descriptions.append(clean)
            else:
                missing['description'] += 1

    elapsed = time.time() - start
    print(f"  Sampled {total_sampled:,} books in {elapsed:.1f}s")

    # Missing value report
    print(f"\n  Missing values (out of {total_sampled:,} sampled books):")
    for field, count in sorted(missing.items()):
        pct = count / total_sampled * 100
        print(f"    {field}: {count:,} ({pct:.1f}%)")

    # --- 3a. Top genres ---
    print("\n  Loading genres...")
    genre_counter = Counter()
    with open(genres_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                genres = obj.get("genres", {})
                for g in genres.keys():
                    for sub_g in g.split(","):
                        genre_counter[sub_g.strip().lower()] += 1
            except json.JSONDecodeError:
                continue

    top_genres = genre_counter.most_common(30)
    fig, ax = plt.subplots(figsize=(12, 7))
    names = [g[0] for g in top_genres]
    counts = [g[1] for g in top_genres]
    bars = ax.barh(range(len(names)), counts, color=sns.color_palette("mako", len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Books")
    ax.set_title("Top 30 Most Frequent Genres")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                f'{count:,}', ha='left', va='center', fontsize=8)
    save_fig(fig, "3a_top_genres")

    # --- 3b. Top authors ---
    print("  Loading authors...")
    author_names = {}
    with open(authors_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                author_names[obj.get("author_id", "")] = obj.get("name", "Unknown")
            except json.JSONDecodeError:
                continue

    # Count books per author from sampled books (re-stream)
    author_counter = Counter()
    with open(books_path, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            if count >= SAMPLE_LIMIT:
                break
            try:
                book = json.loads(line)
                for a in book.get("authors", []):
                    aid = a.get("author_id", "")
                    name = author_names.get(aid, f"ID:{aid}")
                    author_counter[name] += 1
            except json.JSONDecodeError:
                continue
            count += 1

    top_authors = author_counter.most_common(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    a_names = [a[0] for a in top_authors]
    a_counts = [a[1] for a in top_authors]
    ax.barh(range(len(a_names)), a_counts, color=sns.color_palette("rocket", len(a_names)))
    ax.set_yticks(range(len(a_names)))
    ax.set_yticklabels(a_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Books (in sample)")
    ax.set_title("Top 20 Most Prolific Authors")
    save_fig(fig, "3b_top_authors")

    # --- 3c. Language distribution ---
    top_langs = languages.most_common(15)
    fig, ax = plt.subplots(figsize=(10, 5))
    l_names = [l[0] if l[0] else "(empty)" for l in top_langs]
    l_counts = [l[1] for l in top_langs]
    ax.bar(l_names, l_counts, color=PALETTE[:len(l_names)], edgecolor='white')
    ax.set_xlabel("Language Code")
    ax.set_ylabel("Number of Books")
    ax.set_title("Distribution of Book Languages (Top 15)")
    plt.xticks(rotation=45, ha='right')
    save_fig(fig, "3c_language_distribution")

    # --- 3d. Country code distribution ---
    top_countries = countries.most_common(15)
    fig, ax = plt.subplots(figsize=(10, 5))
    c_names = [c[0] for c in top_countries]
    c_counts = [c[1] for c in top_countries]
    ax.bar(c_names, c_counts, color=sns.color_palette("crest", len(c_names)), edgecolor='white')
    ax.set_xlabel("Country Code")
    ax.set_ylabel("Number of Books")
    ax.set_title("Distribution of Book Country Codes (Top 15)")
    plt.xticks(rotation=45, ha='right')
    save_fig(fig, "3d_country_distribution")

    # --- 3e. Average rating distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(avg_ratings, bins=50, color=PALETTE[3], edgecolor='white', linewidth=0.3)
    ax.set_xlabel("Average Rating")
    ax.set_ylabel("Number of Books")
    ax.set_title("Distribution of Book Average Ratings")
    ax.axvline(np.mean(avg_ratings), color='red', linestyle='--', label=f'Mean: {np.mean(avg_ratings):.2f}')
    ax.legend()
    save_fig(fig, "3e_average_rating_distribution")

    # --- 3f. Number of pages distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(num_pages_list, bins=80, color=PALETTE[5], edgecolor='white', linewidth=0.3)
    ax.set_xlabel("Number of Pages")
    ax.set_ylabel("Number of Books")
    ax.set_title("Distribution of Book Page Counts")
    ax.axvline(np.median(num_pages_list), color='red', linestyle='--', label=f'Median: {np.median(num_pages_list):.0f}')
    ax.legend()
    save_fig(fig, "3f_page_count_distribution")

    # --- 3g. Publication year distribution ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pub_years, bins=100, color=PALETTE[6], edgecolor='white', linewidth=0.3)
    ax.set_xlabel("Publication Year")
    ax.set_ylabel("Number of Books")
    ax.set_title("Distribution of Publication Years")
    save_fig(fig, "3g_publication_year_distribution")

    # --- 3h. Word cloud of descriptions ---
    if HAS_WORDCLOUD and descriptions:
        print("  Generating word cloud from book descriptions...")
        all_text = " ".join(descriptions[:20000])  # use up to 20k descriptions
        wc = WordCloud(width=1200, height=600, background_color='white',
                       max_words=200, colormap='viridis', stopwords=None,
                       collocations=False).generate(all_text)
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("Word Cloud of Book Descriptions (Top 200 Terms)")
        save_fig(fig, "3h_description_wordcloud")

    return countries


# ═════════════════════════════════════════════════════════════════════════════
#  4. CULTURAL COVERAGE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def analyze_cultural_coverage(hofstede_path, book_countries):
    print("\n" + "="*80)
    print("  4. CULTURAL COVERAGE ANALYSIS")
    print("="*80)

    df_hof = pd.read_csv(hofstede_path)
    df_hof.columns = [c.strip().lower() for c in df_hof.columns]
    dimensions = ['pdi', 'idv', 'mas', 'uai', 'lto', 'ivr']

    for dim in dimensions:
        df_hof[dim] = pd.to_numeric(df_hof[dim], errors='coerce')
    medians = df_hof[dimensions].median()
    df_hof[dimensions] = df_hof[dimensions].fillna(medians)

    print(f"  Hofstede countries: {len(df_hof)}")
    print(f"  Book country codes found: {len(book_countries)}")

    # --- 4a. Hofstede dimension correlation heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df_hof[dimensions].corr()
    dim_labels = ['Power Distance\n(PDI)', 'Individualism\n(IDV)', 'Masculinity\n(MAS)',
                  'Uncertainty\nAvoidance (UAI)', 'Long-Term\nOrientation (LTO)', 'Indulgence\n(IVR)']
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                xticklabels=dim_labels, yticklabels=dim_labels, mask=mask, ax=ax,
                linewidths=0.5, square=True)
    ax.set_title("Correlation Matrix of Hofstede Cultural Dimensions (119 Countries)")
    plt.xticks(fontsize=8, rotation=30, ha='right')
    plt.yticks(fontsize=8, rotation=0)
    save_fig(fig, "4a_hofstede_correlation_heatmap")

    # --- 4b. PDI vs IDV scatter ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(df_hof['idv'], df_hof['pdi'], c=df_hof['uai'], cmap='RdYlBu_r',
               s=60, edgecolors='grey', linewidth=0.3, alpha=0.8)
    # Label a few notable countries
    notable = ['united states', 'japan', 'nigeria', 'germany', 'brazil', 'china', 'india',
               'south africa', 'united kingdom', 'sweden', 'mexico', 'russia']
    for _, row in df_hof.iterrows():
        if str(row['country']).strip().lower() in notable:
            ax.annotate(str(row['country']).strip(), (row['idv'], row['pdi']),
                        fontsize=7, alpha=0.8, ha='left',
                        xytext=(5, 5), textcoords='offset points')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Uncertainty Avoidance (UAI)")
    ax.set_xlabel("Individualism (IDV)")
    ax.set_ylabel("Power Distance (PDI)")
    ax.set_title("Hofstede Cultural Map: Power Distance vs Individualism")
    save_fig(fig, "4b_pdi_vs_idv_scatter")

    # --- 4c. UAI vs IVR scatter ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(df_hof['ivr'], df_hof['uai'], c=df_hof['lto'], cmap='plasma',
               s=60, edgecolors='grey', linewidth=0.3, alpha=0.8)
    for _, row in df_hof.iterrows():
        if str(row['country']).strip().lower() in notable:
            ax.annotate(str(row['country']).strip(), (row['ivr'], row['uai']),
                        fontsize=7, alpha=0.8, ha='left',
                        xytext=(5, 5), textcoords='offset points')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Long-Term Orientation (LTO)")
    ax.set_xlabel("Indulgence (IVR)")
    ax.set_ylabel("Uncertainty Avoidance (UAI)")
    ax.set_title("Hofstede Cultural Map: Uncertainty Avoidance vs Indulgence")
    save_fig(fig, "4c_uai_vs_ivr_scatter")

    # --- 4d. Hofstede dimension distributions (box plots) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    box_data = [df_hof[d].dropna().values for d in dimensions]
    bp = ax.boxplot(box_data, patch_artist=True, labels=['PDI', 'IDV', 'MAS', 'UAI', 'LTO', 'IVR'])
    colors = sns.color_palette("Set2", 6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Score (0–100)")
    ax.set_title("Distribution of Hofstede Dimension Scores Across 119 Countries")
    save_fig(fig, "4d_hofstede_boxplots")


# ═════════════════════════════════════════════════════════════════════════════
#  5. USER–ITEM INTERACTION ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def analyze_interactions(user_rating_counts, book_rating_counts):
    print("\n" + "="*80)
    print("  5. USER–ITEM INTERACTION ANALYSIS")
    print("="*80)

    user_counts = list(user_rating_counts.values())
    book_counts = list(book_rating_counts.values())

    # --- 5a. Long-tail popularity (log-log) ---
    sorted_book_counts = sorted(book_counts, reverse=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(sorted_book_counts)+1), sorted_book_counts,
            color=PALETTE[1], linewidth=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Book Rank (log scale)")
    ax.set_ylabel("Number of Ratings (log scale)")
    ax.set_title("Long-Tail Distribution of Book Popularity")
    ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Threshold = 5 ratings')
    ax.legend()
    save_fig(fig, "5a_longtail_popularity")

    # --- 5b. Cold-start analysis ---
    thresholds = [1, 2, 3, 5, 10, 20, 50]
    user_cold = {}
    for t in thresholds:
        n_cold = sum(1 for c in user_counts if c <= t)
        pct = n_cold / len(user_counts) * 100
        user_cold[t] = pct
        print(f"  Users with ≤ {t} ratings: {n_cold:,} ({pct:.1f}%)")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(t) for t in thresholds], [user_cold[t] for t in thresholds],
           color=sns.color_palette("flare", len(thresholds)), edgecolor='white')
    ax.set_xlabel("Rating Threshold (≤ N ratings)")
    ax.set_ylabel("% of Users")
    ax.set_title("Cold-Start Analysis: % of Users Below Rating Thresholds")
    for i, t in enumerate(thresholds):
        ax.text(i, user_cold[t] + 0.5, f"{user_cold[t]:.1f}%", ha='center', fontsize=9)
    save_fig(fig, "5b_coldstart_analysis")

    # --- 5c. User activity CDF ---
    sorted_user_counts = np.sort(user_counts)
    cdf = np.arange(1, len(sorted_user_counts)+1) / len(sorted_user_counts)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sorted_user_counts, cdf, color=PALETTE[3], linewidth=1.5)
    ax.set_xscale('log')
    ax.set_xlabel("Number of Ratings (log scale)")
    ax.set_ylabel("Cumulative Proportion of Users")
    ax.set_title("CDF of User Activity (Ratings per User)")
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50th percentile')
    ax.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90th percentile')
    ax.legend()
    save_fig(fig, "5c_user_activity_cdf")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    goodreads_dir = os.path.join(base_dir, "goodreads")
    project_dir = os.path.join(base_dir, "project")
    hofstede_path = os.path.join(project_dir, "hofstede.csv")

    print("╔" + "═"*78 + "╗")
    print("║" + "  GOODREADS + HOFSTEDE — EXPLORATORY DATA ANALYSIS".center(78) + "║")
    print("╚" + "═"*78 + "╝")

    # Phase 1: Dataset profiling
    total_interactions, total_books = profile_datasets(goodreads_dir, hofstede_path)

    # Phase 2: Rating distributions
    user_rating_counts, book_rating_counts = analyze_ratings(goodreads_dir, total_interactions, total_books)

    # Phase 3: Book metadata
    book_countries = analyze_book_metadata(goodreads_dir)

    # Phase 4: Cultural coverage
    analyze_cultural_coverage(hofstede_path, book_countries)

    # Phase 5: Interaction patterns
    analyze_interactions(user_rating_counts, book_rating_counts)

    print("\n" + "="*80)
    print(f"  EDA COMPLETE — All charts saved to: {OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
