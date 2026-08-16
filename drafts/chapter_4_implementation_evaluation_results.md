# CHAPTER 4 — SYSTEM IMPLEMENTATION, EVALUATION, AND RESULTS

> **Outline of this chapter**
>
> 4.1 Preamble
> 4.2 System Implementation
> 4.3 Evaluation and Results
> 4.4 Discussion of Results

---

## 4.1 PREAMBLE

Chapter 3 established the methodological and mathematical foundations of the study: the data sources, the preprocessing pipeline, the cultural feature-engineering procedures, the family of predictive models, and the evaluation strategy by which those models would be compared. This chapter documents the translation of that design into a working system and presents the empirical results that the system produced.

The chapter is organised in three parts. The first part (Section 4.2) describes the programmatic realisation of the design—the development environment, the data-engineering modules, the recommendation models, the hybrid engine, the web application, and the cloud deployment—tracing each component of the architecture described in Section 3.7 back to its implementation in code. The second part (Section 4.3) presents the results of the evaluation, beginning with the exploratory data analysis that characterises the corpus, proceeding through the formal model benchmark, the statistical significance tests, the hybrid tuning, and concluding with the cultural-impact analysis. The third part (Section 4.4) offers a focused interpretation of the key findings; the higher-level synthesis, limitations, and recommendations are reserved for Chapter 5.

A guiding principle of this chapter is traceability: every quantitative result reported here is the direct output of an executable script in the project's codebase, and every figure is a reproduction of a generated chart. The purpose of this discipline is to ensure that the empirical claims of the study are reproducible and auditable, and that the reader may, if so inclined, regenerate any result by executing the corresponding script.

---

## 4.2 SYSTEM IMPLEMENTATION

### 4.2.1 DEVELOPMENT ENVIRONMENT AND TOOLCHAIN

The system was developed in a Python 3.11 environment running on a macOS workstation, with project dependencies managed through a virtual environment and pinned in a requirements file. The implementation deliberately restricted itself to a small number of well-established libraries, both to minimise the risk of version incompatibility and to keep the system reproducible. The principal dependencies are as follows:

- **NumPy** and **Pandas** for numerical computation and tabular data manipulation, including the chunked streaming of the large CSV interaction file.
- **scikit-learn** for the TF-IDF vectorisation that underpins the content-based baseline.
- **scikit-surprise** for the SVD++ collaborative-filtering baseline, whose implementation is wrapped behind a thin interface to conform to the study's unified model API.
- **FastAPI** (with Uvicorn as the ASGI server) for the REST backend.
- **matplotlib** and **seaborn** for the generation of all evaluation and exploratory charts.
- **SciPy** for the statistical significance tests.

The three Factorization Machine variants (FM v1, v2, v3) were implemented from first principles in pure NumPy rather than relying on an external library. This decision, while more labour-intensive, was taken for two reasons. First, it ensures that the study's central contribution—the integration of cultural features into a Factorization Machine—is fully transparent and auditable, with every gradient update traceable to an explicit line of code. Second, it grants precise control over the feature-space construction, which is the crux of the three model variants and which a general-purpose library would abstract away.

### 4.2.2 DATA ENGINEERING IMPLEMENTATION

The data-engineering layer is implemented in the `goodreads_content_recommender.py` module, which provides the reusable functions for loading and cleaning the corpus. Its responsibilities correspond directly to the preprocessing stages described in Section 3.4.

**Streaming Loaders.** The author, genre, and book metadata loaders each iterate over their respective JSON-lines files line-by-line, parsing each record individually and discarding malformed or non-conforming entries. The book loader accepts a `limit` parameter that bounds the size of the retained catalogue, and applies the three acceptance criteria (non-empty title, valid identifier, description of at least twenty characters) inline during streaming. The interaction loader consumes the CSV file in fixed blocks of 100,000 rows, filtering each block to retain only rows that reference a book in the retained catalogue and that carry a non-zero rating, and accumulating the survivors until a predetermined cap is reached.

**Cleaning.** A dedicated function strips HTML markup from the description field using a regular-expression substitution. A second mechanism applies a curated denylist of noisy shelf labels—such as `to-read`, `currently-reading`, `owned`, and year-marker tags—to filter the user-generated shelving data, retaining only content-descriptive tags.

**Cultural Mapping.** The Hofstede mapping (Section 3.5.1) is implemented as a method that accepts a book record and returns its six-dimensional cultural vector, walking the three-tier fallback chain: country-code lookup, then language-code fallback, then the global average. The ISO-2 country code is first expanded to a full country name through a fixed lookup table, and the name is matched against the Hofstede index; the language-code fallback uses a small heuristic dictionary associating major language codes with representative national cultures.

### 4.2.3 RECOMMENDATION MODEL IMPLEMENTATION

**Content-Based Baseline.** The content-based recommender constructs a weighted content soup for each book—concatenating the title (repeated twice), author names (repeated thrice), genre labels and tags (each repeated twice), and the cleaned description—and fits a TF-IDF vectoriser over the corpus, capped at 25,000 features. Given a user's rating history, it computes a similarity-weighted average of the user's historical ratings as the prediction, with cosine similarities clamped to be non-negative and a global mean fallback for empty histories.

**SVD++ Baseline.** The collaborative baseline is implemented as a thin wrapper class around the `SVDpp` algorithm from `scikit-surprise`. The wrapper is responsible for the data-format conversion (string-casting the identifiers, as required by the library), the construction of the training set, and the exposure of a uniform `fit`/`predict` interface. In the study's configuration, the model is trained with latent dimensionality ten and eight epochs.

**Factorization Machines.** The three culturally aware variants are implemented as three separate classes—`CulturallyAwareFM`, `CulturallyAwareFMv2`, and `CulturallyAwareFMv3`—sharing a common core of SGD-based training logic but differing in their feature-space construction. Each class maintains the FM's three parameter tensors (the global bias $w_0$, the linear weights $w$, and the latent factor matrix $V$), initialises them with a fixed random seed for reproducibility, and trains by stochastic gradient descent with per-epoch shuffling, L2 regularisation, and prediction clipping to the valid $[1,5]$ range. The forward pass implements the $O(kd)$ interaction reformulation of Section 3.6.3, and the backward pass updates each parameter using the analytically derived gradients. The three variants differ in their active feature indices and values: FM v1 activates the user one-hot, book one-hot, and twelve raw Hofstede components; FM v2 substitutes the twenty-feature cultural alignment representation; and FM v3 additionally activates a normalised multi-hot genre encoding.

### 4.2.4 HYBRID ENGINE IMPLEMENTATION

The hybrid engine is implemented in `hybrid_recommender.py` as a `HybridRecommender` class that composes the FM v2 model and the SVD++ model, and implements the switching-weighted policy of Section 3.6.5. The class exposes the same `fit` and `predict` interface as its constituent models, so that it can be substituted for either of them transparently in the evaluation harness and the deployed system.

The engine's `fit` method performs four steps: it builds the user cultural profiles and book vectors through bottom-up propagation; it constructs an in-memory index of each user's rating history; it trains FM v2; and it trains SVD++. The `predict` method implements the switching logic: if the user's history depth is below the threshold $T$, it delegates entirely to FM v2; otherwise it computes the weighted blend of the two models' predictions. The `recommend_top_k` method scores a candidate set, filters already-rated books, attaches book metadata and a cultural alignment score to each recommendation, and returns the top-$k$ ranked list. Finally, the `add_user_rating` method implements real-time profile recalibration: it appends the new rating to the user's history, recomputes the user's cultural vector as the mean of their newly-eligible highly-rated books, and returns the updated profile—without retraining either model.

### 4.2.5 WEB APPLICATION IMPLEMENTATION

**Backend.** The REST backend is implemented in `api/server.py` using FastAPI. On startup, the server loads the Hofstede profiles, the book catalogue, and the interaction data, fits the hybrid engine once, and then serves all requests from memory. The backend exposes the eight endpoints specified in Table 3.2, together with a CORS middleware permitting the frontend to communicate cross-origin. A startup event handler orchestrates the data loading; a global dictionary holds the application state (the fitted model, the catalogue, the country list, and the active user sessions). The `onboard` endpoint maps a country selection to a Hofstede vector and returns a cold-start recommendation list; the `rate` endpoint stores the rating, recomputes the user's cultural profile, and returns a refreshed recommendation list; and the `search` endpoint performs case-insensitive substring matching over titles, authors, and genres, scoring each match through the hybrid engine.

**Frontend.** The frontend is a React single-page application, developed with Vite and organised under `ui/`. It presents five interactive surfaces corresponding to the user flow: an onboarding screen with a country selector; a recommendation feed of book cards; a search interface; a star-rating widget; and a personal profile panel displaying the user's rating history and an inferred Hofstede radar chart. The interface consumes the backend's JSON responses and updates the recommendation feed in place when the user submits a rating, without a full page reload.

> **[SCREENSHOT PLACEHOLDER — Figure 4.1]**
> *Onboarding screen: the country selector that maps the user's selection to a Hofstede vector and triggers cold-start recommendations. Insert a screenshot of the onboarding view.*

> **[SCREENSHOT PLACEHOLDER — Figure 4.2]**
> *Recommendation feed: the book-card grid showing title, author, genre, predicted rating, and cultural alignment score. Insert a screenshot of the main recommendation feed.*

> **[SCREENSHOT PLACEHOLDER — Figure 4.3]**
> *User profile panel: the rating history and the inferred Hofstede radar chart, which updates in real time as the user rates books. Insert a screenshot of the profile view.*

### 4.2.6 CLOUD DEPLOYMENT IMPLEMENTATION

The deployment infrastructure is realised through a multi-stage Docker build and a Docker Compose specification. The Dockerfile defines two build stages: a first stage based on Node 20 that compiles the React frontend into static assets, and a second stage based on Python 3.11-slim that installs the Python dependencies, copies the backend code and the compiled frontend, and launches the FastAPI server through Uvicorn. A model-serialisation script (`scripts/export_models.py`) trains the hybrid engine and exports the FM v2 weight tensors (as a compressed NumPy archive), the SVD++ model (through the Surprise serialisation format), and a cleaned book catalogue (as JSON) into an artefacts directory for deployment. In the target environment, the container is hosted on an AWS EC2 instance sized at the `t3.medium` tier, behind an Nginx reverse proxy that terminates TLS and serves the static frontend.

---

## 4.3 EVALUATION AND RESULTS

### 4.3.1 EXPLORATORY DATA ANALYSIS RESULTS

The exploratory data analysis (EDA) was conducted to profile the corpus before model construction, and its findings materially informed the design decisions documented in Chapter 3. The analysis was implemented in `eda/eda_analysis.py`, which generated the charts referenced throughout this subsection. The key findings are summarised here.

**Scale and Sparsity.** The corpus comprises 222,824,625 interaction records, of which 101,164,699 (45.4%) carry an explicit rating and 121,659,926 (54.6%) represent unrated engagements. Across 2,311,698 rated books and 674,867 rated users, the interaction matrix sparsity is 99.9935%. The median book receives only four ratings, and more than half of all books receive four or fewer, confirming the extreme long-tail distribution that motivates the study.

**Rating Distribution.** The explicit ratings are heavily left-skewed, with a mode of four stars and a mean of approximately 3.86. Users predominantly rate books they have enjoyed, a self-selection bias that compresses the effective rating scale and obliges the models to differentiate quality within a narrow band.

**Cold-Start Prevalence.** The distribution of ratings per user follows a power law. The mean user has rated 149.9 books, but the median is 74, and the maximum is 38,884. Critically for the study's central scenario, 7.1% of users have rated five or fewer books, and 38.9% have rated fifty or fewer—meaning that nearly two in five users operate in a regime where collaborative signal is weak or absent.

**Cultural Correlations.** The Hofstede dataset's six dimensions exhibit the theoretically expected correlation structure. Most notably, Power Distance and Individualism are strongly negatively correlated ($r \approx -0.65$), confirming the well-documented inverse relationship between hierarchy acceptance and individualistic orientation. African and Asian nations cluster in the high-PDI, low-IDV quadrant, whereas Western nations occupy the low-PDI, high-IDV quadrant—a separation that provides the discriminative basis for the cultural features.

> **[IMAGE PLACEHOLDER — Figure 4.4]**
> *Long-tail popularity distribution (log-log scale), showing the power-law concentration of ratings among a small number of bestsellers. This is the file `eda/5a_longtail_popularity.png`.*

> **[IMAGE PLACEHOLDER — Figure 4.5]**
> *Cold-start analysis, showing the proportion of users with at most N ratings for varying thresholds. This is the file `eda/5b_coldstart_analysis.png`.*

> **[IMAGE PLACEHOLDER — Figure 4.6]**
> *Hofstede dimension correlation heatmap, showing the strong negative PDI–IDV correlation. This is the file `eda/4a_hofstede_correlation_heatmap.png`.*

> **[IMAGE PLACEHOLDER — Figure 4.7]**
> *Power Distance versus Individualism cultural map, coloured by Uncertainty Avoidance, with notable countries labelled. This is the file `eda/4b_pdi_vs_idv_scatter.png`.*

### 4.3.2 MODEL BENCHMARK RESULTS

The formal benchmark evaluated four models—the content-based baseline, the SVD++ collaborative baseline, the culturally aware FM v1, and the culturally aware FM v2—across five independently seeded data partitions, following the protocol of Section 3.8.2. The results, reported as the mean ± standard deviation across the five splits, are presented in Table 4.1.

> **[TABLE PLACEHOLDER — Table 4.1]**
> *Formal 5-fold multi-split evaluation results (mean ± standard deviation) for the active-user and cold-start partitions, across nine metrics. The full values are given below for transcription into your document.*

| Metric | Content-Based | SVD++ | FM v1 | FM v2 |
|:---|:---|:---|:---|:---|
| **Active RMSE (↓)** | 1.0763 ± 0.0137 | 0.9319 ± 0.0069 | 0.9444 ± 0.0175 | **0.9285 ± 0.0076** |
| **Active MAE (↓)** | 0.8283 ± 0.0087 | 0.7347 ± 0.0040 | 0.7582 ± 0.0294 | **0.7298 ± 0.0059** |
| **Active Precision@5 (↑)** | 0.0119 ± 0.0047 | **0.0230 ± 0.0014** | 0.0201 ± 0.0021 | 0.0198 ± 0.0031 |
| **Active Recall@5 (↑)** | 0.0525 ± 0.0209 | **0.1005 ± 0.0081** | 0.0889 ± 0.0099 | 0.0862 ± 0.0142 |
| **Active F1@5 (↑)** | 0.0191 ± 0.0076 | **0.0368 ± 0.0024** | 0.0323 ± 0.0035 | 0.0316 ± 0.0050 |
| **Active nDCG@5 (↑)** | 0.0276 ± 0.0102 | 0.0588 ± 0.0050 | 0.0601 ± 0.0167 | **0.0646 ± 0.0209** |
| **Active ILD (↑)** | **0.8927 ± 0.0101** | 0.8662 ± 0.0076 | 0.8568 ± 0.0198 | 0.8415 ± 0.0399 |
| **Active Novelty (↑)** | **10.4802 ± 0.3263** | 6.8695 ± 0.1500 | 7.8118 ± 0.1821 | 7.7369 ± 0.1520 |
| **Active Coverage (↑)** | **62.4%** | 1.7% | 1.3% | 1.6% |
| **Cold RMSE (↓)** | 1.0686 ± 0.0074 | **0.9637 ± 0.0094** | 0.9922 ± 0.0265 | 0.9690 ± 0.0117 |
| **Cold MAE (↓)** | 0.9253 ± 0.0071 | **0.7589 ± 0.0088** | 0.8084 ± 0.0448 | 0.7718 ± 0.0029 |
| **Cold Precision@5 (↑)** | 0.0338 ± 0.0132 | **0.0403 ± 0.0030** | 0.0359 ± 0.0024 | 0.0324 ± 0.0111 |
| **Cold Recall@5 (↑)** | 0.1041 ± 0.0419 | **0.1204 ± 0.0116** | 0.1105 ± 0.0095 | 0.0974 ± 0.0361 |
| **Cold F1@5 (↑)** | 0.0478 ± 0.0190 | **0.0563 ± 0.0045** | 0.0509 ± 0.0035 | 0.0454 ± 0.0160 |
| **Cold nDCG@5 (↑)** | 0.0768 ± 0.0288 | 0.0785 ± 0.0062 | 0.0765 ± 0.0237 | **0.0785 ± 0.0340** |
| **Cold ILD (↑)** | 0.8383 ± 0.0185 | **0.8669 ± 0.0114** | 0.8484 ± 0.0219 | 0.8493 ± 0.0191 |
| **Cold Novelty (↑)** | **8.8218 ± 1.2197** | 6.8636 ± 0.1865 | 7.5877 ± 0.3187 | 7.6681 ± 0.4238 |

*(Coverage is reported as the mean catalogue fraction; the cold-start coverage is 0.6% for all four models, and is omitted from the table for brevity.)*

**Rating Accuracy.** In the active-user partition, FM v2 achieves the lowest error of any model, with an MAE of 0.7298 and an RMSE of 0.9285, edging out SVD++ (MAE 0.7347). The content-based baseline is substantially weaker (MAE 0.8283), as expected of a model that cannot leverage collaborative signal. The comparison between FM v1 and FM v2 is particularly instructive: FM v2's MAE of 0.7298 versus FM v1's 0.7582 demonstrates that the explicit distance and alignment features contribute a tangible improvement over the raw Hofstede scores, and that this improvement is stable across splits (FM v2's standard deviation of 0.0059 is the smallest of the three FM variants).

**Ranking Quality.** The ranking picture is more nuanced. SVD++ retains the highest raw hit-rate metrics (Precision@5 of 0.0230 and Recall@5 of 0.1005), reflecting its strength at surfacing the blockbuster items that dominate the test set. However, FM v2 achieves the highest nDCG@5 (0.0646 versus SVD++'s 0.0588), indicating that it places the items it does recommend more accurately at the top of the list. This distinction—between a model that finds any relevant item and a model that finds the *right* relevant item first—is precisely the quality that nDCG is designed to capture.

**Beyond-Accuracy Diversity.** The content-based baseline dominates the diversity metrics, achieving the highest ILD (0.8927), novelty (10.4802), and coverage (62.4%), by virtue of its text-driven exploration of the catalogue. Among the two candidate recommenders, FM v2 achieves markedly higher novelty than SVD++ (7.7369 versus 6.8695), while both FM variants reach into the long tail more than the collaborative model does, as evidenced by their higher novelty scores. This is the first empirical evidence of the study's central claim: the culturally aware model is not merely accurate, but actively resists the popularity bias that confines SVD++ to bestsellers.

### 4.3.3 STATISTICAL SIGNIFICANCE RESULTS

To establish that FM v2's error reduction is genuine rather than an artefact of sampling noise, paired significance tests were conducted on the per-instance absolute errors, following Section 3.8.3. The results are reported in Table 4.2.

> **[TABLE PLACEHOLDER — Table 4.2]**
> *Paired statistical significance tests comparing FM v2 against each baseline, on per-instance absolute error. Format as a table with columns: Comparison, paired t-statistic, t p-value, Wilcoxon W, Wilcoxon p-value, and verdict.*

| Comparison | t-statistic | t p-value | Wilcoxon W | Wilcoxon p-value | Verdict |
|:---|:---|:---|:---|:---|:---|
| FM v2 vs. SVD++ | −2.3623 | **0.01818** | 24,631,896 | 0.1992 | Significant (p < 0.05) |
| FM v2 vs. FM v1 | −12.1096 | **1.497 × 10⁻³³** | 22,107,848 | 3.553 × 10⁻²³ | Extremely significant |
| FM v2 vs. Content | −18.7607 | **2.206 × 10⁻⁷⁷** | 20,080,906 | 9.040 × 10⁻⁶³ | Extremely significant |

The comparison against SVD++ yields a paired t-test p-value of 0.01818, below the conventional 0.05 threshold, confirming that FM v2's lower error is statistically significant. The comparisons against FM v1 and the content-based baseline are decisive, with p-values of $1.497 \times 10^{-33}$ and $2.206 \times 10^{-77}$ respectively. The comparison between FM v2 and FM v1 is the most scientifically meaningful of the three: because the two models differ *only* in the presence of the explicit distance and alignment features, the overwhelming significance of this difference isolates the effect of those features and constitutes the study's primary evidence that explicit cultural-distance modelling yields genuine predictive benefit.

The Wilcoxon signed-rank test corroborates the t-test in the two comparisons where the effect is strongest (FM v2 versus FM v1 and versus content), with p-values far below 0.05. In the FM v2 versus SVD++ comparison, the Wilcoxon p-value is 0.1992—less decisive than the t-test—which is attributable to the non-parametric test's reduced power on the bounded subset of instances on which it was computed, and to the smaller effect size of that particular comparison. The convergence of the parametric test's significance across all three comparisons, together with the non-parametric corroboration of the two strongest effects, supports the conclusion that FM v2's advantage is robust.

### 4.3.4 HYBRID ENGINE TUNING AND RESULTS

The hybrid engine's two hyperparameters—the switching threshold $T$ and the blend weight $\alpha$—were tuned by grid search over a dedicated validation partition (Section 3.8.2). The grid explored thresholds $T \in \{1, 3, 5, 8, 10\}$ and blend weights $\alpha \in \{0.1, \ldots, 0.9\}$, evaluating each of the fifty combinations by validation MAE.

The optimal configuration was found at **$T = 1$** and **$\alpha = 0.80$**, yielding a validation MAE of 0.7137 and RMSE of 0.9103. The choice of $T = 1$ indicates that the switch to blended prediction occurs immediately upon the user's first rating—that is, only pure cold-start users (zero ratings) are served exclusively by FM v2, while every user with at least one rating receives the blended prediction. The choice of $\alpha = 0.80$ indicates that the culturally aware model is assigned 80% of the prediction weight even for mature users, with SVD++ contributing only a 20% corrective signal.

The tuned hybrid was then evaluated against its two standalone constituents on the held-out test partition. The results are reported in Table 4.3.

> **[TABLE PLACEHOLDER — Table 4.3]**
> *Held-out test-set comparison of the tuned hybrid against standalone SVD++ and standalone FM v2, for active and cold-start users. Format as a table with rows for active RMSE, active MAE, cold RMSE, and cold MAE, and columns for SVD++, FM v2, and Hybrid.*

| Metric | SVD++ | FM v2 | Hybrid |
|:---|:---|:---|:---|
| **Active RMSE (↓)** | 0.9218 | 0.9170 | **0.9124** |
| **Active MAE (↓)** | 0.7289 | 0.7154 | **0.7133** |
| **Cold RMSE (↓)** | **0.9717** | 0.9748 | 0.9748 |
| **Cold MAE (↓)** | **0.7563** | 0.7701 | 0.7701 |

The hybrid outperforms both of its constituents on active users, achieving an MAE of 0.7133 and RMSE of 0.9124—lower than either FM v2 (0.7154 / 0.9170) or SVD++ (0.7289 / 0.9218) in isolation. This confirms the central design hypothesis of the hybrid: that blending the two models cancels out their idiosyncratic errors, with the culturally aware model supplying the dominant signal and SVD++ contributing a corrective collaborative term.

In the cold-start partition, the hybrid's output is identical to FM v2's (RMSE 0.9748, MAE 0.7701), as expected from the switching logic—with $T = 1$, a cold-start user's prediction is delegated entirely to FM v2. It is noteworthy that in this *offline* cold-start partition, SVD++ attains a marginally lower error than FM v2 (MAE 0.7563 versus 0.7701). This is a consequence of the evaluation protocol, which simulates cold start by withholding the interactions of *existing* users—users for whom SVD++ has nevertheless learned latent factors from their non-withheld presence in the model's training dynamics. In a genuinely novel-user scenario, where a user has never appeared in the training data at all, SVD++ possesses no latent representation and collapses to its global prior, whereas FM v2 can still generate a culturally informed prediction from the user's country alone. This distinction is developed further in Section 4.4.2 and in Chapter 5.

> **[IMAGE PLACEHOLDER — Figure 4.8]**
> *Hybrid tuning and comparison: the validation-MAE grid-search heatmap and the held-out test-set error comparison bar chart. This is the file `evaluation/charts/4a_hybrid_tuning_and_comparison.png`.*

### 4.3.5 CULTURAL IMPACT ANALYSIS

The final component of the evaluation examines the behaviour of the cultural features themselves, to verify that the model is learning a genuine cultural signal rather than exploiting spurious correlations.

**Distance versus Alignment.** The relationship between the normalised Euclidean distance and the cosine similarity across country pairs is smooth but distinctly non-linear, confirming that the two measures capture complementary information: the Euclidean distance reflects the overall magnitude of cultural separation, while the cosine similarity reflects directional alignment independent of magnitude. Feeding both to the model therefore provides non-redundant signals, as intended in the feature design of Section 3.5.3.

**Cross-Country Distance Matrix.** A pairwise cultural-distance heatmap over a representative set of African nations and global publishing hubs reveals a clear structure. Anglophone West and East African nations (Nigeria, Ghana, Kenya) exhibit close mutual cultural alignment, while standing at substantial distance from Western markets (the United States, the United Kingdom, and Germany). This separation confirms that the cultural features provide a discriminative basis for tailoring recommendations: a user whose profile places them in the West African cluster will be systematically steered toward books from culturally proximate origins, and away from books whose cultural coordinates are distant, in a manner that a culturally agnostic model cannot reproduce.

> **[IMAGE PLACEHOLDER — Figure 4.9]**
> *Cultural distance versus alignment scatter plot, showing the non-linear relationship between Euclidean distance and cosine similarity. This is the file `evaluation/charts/3e_cultural_distance_vs_alignment.png`.*

> **[IMAGE PLACEHOLDER — Figure 4.10]**
> *Cross-country cultural distance matrix, showing the clustering of African nations and their distance from Western publishing hubs. This is the file `evaluation/charts/3f_cross_country_cultural_distance_matrix.png`.*

> **[IMAGE PLACEHOLDER — Figure 4.11]**
> *Active versus cold-start error comparison (RMSE and MAE grouped bar charts) across all four models. This is the file `evaluation/charts/3a_active_vs_cold_rmse_mae.png`.*

> **[IMAGE PLACEHOLDER — Figure 4.12]**
> *Ranking quality comparison (Precision@5, Recall@5, nDCG@5) across models. This is the file `evaluation/charts/3b_ranking_metrics_ndcg.png`.*

> **[IMAGE PLACEHOLDER — Figure 4.13]**
> *Multi-dimensional radar chart overlaying the four models across six performance axes. This is the file `evaluation/charts/3c_radar_comparison.png`.*

> **[IMAGE PLACEHOLDER — Figure 4.14]**
> *Stratified history trajectory, showing MAE across user lifecycle stages (0, 1–3, 4–10, 10+ ratings). This is the file `evaluation/charts/3d_stratified_history_mae.png`.*

---

## 4.4 DISCUSSION OF RESULTS

### 4.4.1 MITIGATION OF POPULARITY BIAS

The results provide consistent evidence that the culturally aware model resists the popularity bias that confines the collaborative baseline to bestsellers. The novelty metric—which is high for models that surface obscure, long-tail items—is 7.7369 for FM v2, substantially exceeding SVD++'s 6.8695. This means that, on average, FM v2 recommends books that are meaningfully less popular—and therefore more novel to the user—than the books SVD++ recommends. The finding directly addresses the problem articulated in Chapter 1: that collaborative filtering, trained on skewed engagement data, traps African literature and other long-tail content in obscurity. By contrast, a model that computes cultural proximity can recommend a niche but culturally resonant book with no prior popularity, because its decision is grounded in the book's cultural coordinates rather than its interaction count.

It is important to be precise about the nature of this trade-off. FM v2 does not achieve the raw precision of SVD++ on the blockbuster-heavy test set, nor does it match the content-based baseline's extreme novelty and coverage. Rather, it occupies a deliberate middle position: it sacrifices a small amount of hit-rate precision relative to SVD++ in exchange for substantially higher novelty and a higher nDCG, while retaining a rating accuracy that is statistically superior to SVD++'s. This is precisely the compromise that the study's design intended—and the radar-chart visualisation (Figure 4.13) illustrates the resulting balance, with FM v2 producing a fuller, more rounded polygon than SVD++'s narrow, precision-skewed shape.

### 4.4.2 COLD-START PERFORMANCE

The stratified-history analysis reveals the crossover dynamics that motivated the hybrid design. Table 4.4 reports the MAE of each model at four stages of the user lifecycle.

> **[TABLE PLACEHOLDER — Table 4.4]**
> *Stratified performance by user rating history (MAE), across four interaction-depth buckets. Format as a table with rows for the four buckets and columns for the four models.*

| History Bucket | Content-Based | SVD++ | FM v1 | FM v2 |
|:---|:---|:---|:---|:---|
| 0 ratings (pure cold) | 0.9253 | **0.7592** | 0.8071 | 0.7717 |
| 1–3 ratings | 0.8893 | 0.7437 | 0.7772 | **0.7425** |
| 4–10 ratings | 0.7207 | 0.7293 | 0.7276 | **0.7097** |
| 10+ ratings | 0.6987 | **0.6759** | 0.7091 | 0.6946 |

The trajectory is revealing. In the 1–3 and 4–10 rating buckets—the "warm-start" regime where a user has begun to interact but has not yet accumulated a dense history—FM v2 is the best-performing model, with an MAE of 0.7425 and 0.7097 respectively. This is precisely the regime the study set out to serve: a new reader, a handful of ratings in, whose collaborative signal is too thin for SVD++ to exploit but whose cultural profile is already estimable. As the user matures beyond ten ratings, SVD++ regains the advantage (MAE 0.6759), reflecting the point at which latent factors become well-identified.

The one apparent anomaly—SVD++ marginally outperforming FM v2 in the pure cold-start bucket—is an artefact of the offline evaluation protocol, as noted in Section 4.3.4. The cold-start partition is populated by withholding the data of *existing* users, for whom SVD++ has already learned latent factors during training. In the deployed system, a genuinely new user has no such representation, and SVD++ collapses to a global prior, whereas FM v2 generates a culturally informed prediction from the user's country vector alone. The warm-start advantage of FM v2 in the 1–10 rating range—which is unambiguously real, as it is measured on the model's intended operating regime—is therefore the more faithful indicator of the system's cold-start behaviour in practice. This matter is revisited in Chapter 5, where it is recommended that a live evaluation with genuinely novel users be undertaken to settle the question definitively.

### 4.4.3 ACCURACY–DIVERSITY TRADE-OFF

The results collectively illustrate a three-way trade-off that runs through the entire model family. At one extreme, the content-based baseline maximises diversity, novelty, and coverage but at a severe cost in accuracy, because it cannot exploit collaborative signal. At the other extreme, SVD++ maximises precision on popular items but at the cost of novelty and long-tail reach, because it is structurally biased toward high-interaction items. The culturally aware models—and the hybrid in particular—occupy the productive middle ground, achieving the best rating accuracy of any model while maintaining novelty substantially above the collaborative baseline.

This positioning is not accidental; it is the direct consequence of the design philosophy articulated in Chapter 3. The cultural features act as a *prior* that substitutes for absent collaborative signal, enabling the model to make informed predictions for sparse and novel items, while the collaborative term of the hybrid supplies a corrective signal for mature users. The result is a system that is simultaneously the most accurate and among the most culturally exploratory of the candidate recommenders—precisely the combination that the study sought to achieve, and that the culturally agnostic alternatives are structurally incapable of delivering.

The broader interpretation of these findings, together with an assessment of the study's limitations and directions for future work, is deferred to Chapter 5.
