# CHAPTER 3 — METHODOLOGY

> **Outline of this chapter**
>
> 3.1 Preamble
> 3.2 Research Design
> 3.3 Data Requirements and Sources
> 3.4 Data Preprocessing
> 3.5 Cultural Feature Engineering
> 3.6 Model Development
> 3.7 System Architecture
> 3.8 Evaluation Strategy
>
> *(Sections are written incrementally; completed sections appear below.)*

---

## 3.1 PREAMBLE

This chapter presents the methodological framework underpinning the design, development, and evaluation of the culturally aware hybrid recommendation platform described in the preceding chapters. The research adopts a quantitative, experimental approach in which a recommendation engine is constructed through a structured machine-learning pipeline and subsequently subjected to rigorous empirical validation. The methodology is organised to reflect the natural lifecycle of a data-driven system: the sourcing and preparation of data, the engineering of culturally informed features, the specification and training of predictive models, the integration of those models into a deployable software architecture, and finally the evaluation of the system against established and novel performance criteria.

The central methodological challenge addressed in this chapter is the translation of an abstract sociological theory—Hofstede's Cultural Dimensions—into computable, quantitative features that can be fed into machine-learning models. This translation is non-trivial: cultural identity is inherently qualitative, relational, and multidimensional, whereas recommendation algorithms operate on numerical vectors. Accordingly, this chapter dedicates significant attention to the feature-engineering procedures by which geographic and linguistic metadata are mapped onto six-dimensional cultural vectors, and by which those vectors are subsequently transformed into a set of twenty alignment and distance features that encode both the magnitude and the direction of cultural proximity between a user and a book.

A second defining characteristic of the methodology is its comparative and adversarial posture. Rather than developing a single model in isolation, the research constructs a family of predictive models—ranging from a content-based baseline and a collaborative-filtering baseline, through three variants of a culturally aware Factorization Machine, to a Switching-Weighted Hybrid engine—and evaluates them against one another under controlled conditions. This comparative design is essential to the study's validity, because the central research claim is not merely that a culturally aware model performs well in absolute terms, but that the integration of cultural dimensions produces a *measurable, statistically significant improvement* over culturally agnostic alternatives. The evaluation strategy is therefore constructed to isolate the effect of the cultural features specifically, using paired statistical significance tests and stratified analysis across different levels of user interaction history.

Throughout this chapter, the methodology is described with reference to the actual artefacts of the implemented system—the code modules, the data files, and the evaluation scripts—so that the account remains faithful to what was built rather than an idealised description of what might have been. Where the methodology admits design choices, the rationale for each choice is made explicit, together with the consequences of the alternative that was rejected. This commitment to traceability ensures that the empirical results reported in Chapter 4 can be interpreted in the context of the precise procedures that produced them.

---

## 3.2 RESEARCH DESIGN

### 3.2.1 RESEARCH PARADIGM

The research is situated within the positivist tradition of computer science and, more specifically, within the paradigm of **computational experimentation**. The object of study—the behaviour of a recommendation algorithm—is treated as a deterministic, measurable artefact whose properties can be quantified through controlled manipulation of its inputs and observation of its outputs. This paradigm is appropriate for the research problem for two reasons. First, the research question is fundamentally evaluative: it seeks to establish whether the integration of cultural dimensions improves predictive accuracy and recommendation quality relative to baseline methods. Such questions are answerable only through empirical measurement. Second, the recommendation problem admits of a clear, well-defined target—the user's numeric rating—which provides an objective ground truth against which model performance can be assessed.

### 3.2.2 NATURE OF THE STUDY

The study is **quantitative** and **experimental** in nature. It does not rely on surveys, interviews, or qualitative user studies; rather, it derives all of its conclusions from the statistical analysis of a large, real-world dataset of user–book interactions. The quantitative orientation is justified by the scale of the underlying data (in excess of 222 million interactions in the raw corpus) and by the need for replicable, objective evaluation metrics that permit rigorous comparison between models.

Within this quantitative framework, the study adopts a **comparative experimental design**. A set of candidate models, spanning three distinct families of recommendation technique, is trained on an identical training partition and evaluated on an identical test partition using an identical metric suite. This design permits direct attribution of any observed performance differential to the specific properties of each model, and in particular to the presence or absence of cultural features. The comparative design is operationalised through a multi-split cross-validation procedure (Section 3.8.2) that averages results over five independently seeded data partitions, thereby mitigating the risk that any observed difference is an artefact of a particular train/test split.

### 3.2.3 RESEARCH PIPELINE OVERVIEW

The overall research design is organised as a six-stage pipeline, illustrated in Figure 3.1. Each stage corresponds to a discrete section of this chapter, and the output of each stage forms the input to the next.

```
┌───────────────┐    ┌────────────────┐    ┌───────────────────┐
│  Data         │    │  Data          │    │  Cultural Feature │
│  Sourcing     │───▶│  Preprocessing │───▶│  Engineering      │
└───────────────┘    └────────────────┘    └───────────────────┘
                                                   │
                                                   ▼
┌───────────────┐    ┌────────────────┐    ┌───────────────────┐
│  Evaluation   │◀───│  Model         │◀───│  Model            │
│  & Analysis   │    │  Integration   │    │  Development      │
└───────────────┘    └────────────────┘    └───────────────────┘
```

> **[IMAGE PLACEHOLDER — Figure 3.1]**
> *Research pipeline overview: data sourcing → preprocessing → cultural feature engineering → model development → model integration → evaluation. Recreate this as a horizontal six-stage flowchart with arrows; the ASCII sketch above gives the exact node labels and flow.*

**Stage 1 — Data Sourcing** (Section 3.3): acquisition of the UCSD Goodreads corpus for book metadata and user–item interactions, together with the Hofstede cultural dimensions dataset. This stage establishes the empirical foundation of the study.

**Stage 2 — Data Preprocessing** (Section 3.4): streaming-based parsing of the large JSON and CSV files, alignment of book and interaction identifiers, cleaning of text fields, and handling of missing values. This stage transforms raw files into structured, analysis-ready tables.

**Stage 3 — Cultural Feature Engineering** (Section 3.5): the mapping of geographic and linguistic metadata onto six-dimensional Hofstede vectors, the bottom-up propagation of book-level cultural vectors into user-level cultural profiles, and the derivation of twenty continuous alignment and distance features. This stage is the methodological heart of the research, as it is here that sociological theory is rendered computationally tractable.

**Stage 4 — Model Development** (Section 3.6): the specification and training of the model family—the content-based baseline, the collaborative-filtering baseline (SVD++), three variants of the culturally aware Factorization Machine, and the Switching-Weighted Hybrid engine that unifies them.

**Stage 5 — Model Integration** (Section 3.7): the embedding of the trained models into a deployable software architecture comprising a FastAPI backend, a React single-page application frontend, and a containerised cloud deployment topology.

**Stage 6 — Evaluation** (Section 3.8): the measurement of each model's performance using a multi-metric evaluation suite, the statistical testing of performance differentials, and the benchmarking of the culturally aware models against the traditional baselines. The outputs of this stage are reported and interpreted in Chapter 4.

### 3.2.4 JUSTIFICATION OF THE EXPERIMENTAL APPROACH

The experimental approach is preferred over alternative designs for the following reasons. First, an experimental design enables the **controlled isolation of the cultural dimension variable**, which is the crux of the research contribution. By holding the dataset, the training procedure, and the evaluation protocol constant while varying only the presence of cultural features, the design permits a clean causal claim about the effect of cultural awareness on recommendation quality. Second, the approach is **reproducible**: all models, scripts, and procedures are implemented in code and executed deterministically with fixed random seeds, so that the results reported in Chapter 4 can be regenerated. Third, the approach is **scalable** to the size of the dataset, relying on streaming and chunked processing techniques that avoid the memory constraints that would preclude the analysis of a multi-gigabyte corpus on commodity hardware.

It should, however, be acknowledged that the experimental approach carries an inherent limitation: the evaluation is conducted *offline*, against historical rating data, rather than in a live deployment with real users. This limitation—which is characteristic of the recommender-systems literature at large and has been identified as a pervasive methodological weakness by Klimashevskaia et al. (2024)—is explicitly acknowledged in the scope and limitations of this study and is revisited in the discussion chapter.

---

## 3.3 DATA REQUIREMENTS AND SOURCES

This section specifies the data that the recommendation system requires in order to fulfil its two complementary functions: predicting a user's rating for a given book, and ranking candidate books for recommendation. It begins by defining the three logically distinct categories of data required by the system, and then describes the two concrete data sources—the UCSD Goodreads corpus and the Hofstede cultural dimensions dataset—from which those categories are satisfied. A summary of the datasets and their scale is provided in Table 3.1.

### 3.3.1 REQUIRED DATA COMPONENTS

The recommendation framework described in this study is a hybrid system that combines collaborative filtering, content-based analysis, and culturally aware modelling. Consequently, the system requires three distinct categories of data, each of which is indispensable to a particular component of the model.

**Component 1 — User–Item Interaction Data.** The collaborative-filtering component (SVD++) and the culturally aware Factorization Machines both operate on historical records of the form *(user, book, rating)*, where the rating is an explicit numeric score on a five-point scale. This interaction data serves two purposes. First, it provides the supervised learning signal—the ground-truth target variable—against which all predictive models are trained and evaluated. Second, the density and distribution of interactions determine the severity of the cold-start and sparsity problems that the culturally aware model is specifically designed to mitigate. The interaction data must therefore include both the rated book identifiers and the user identifiers, so that the two can be linked to their respective side information (book metadata and cultural profiles).

**Component 2 — Book Metadata.** The content-based component and the cultural feature-engineering process both require descriptive metadata for each book. For the content-based baseline, this metadata consists of the textual fields—title, description, authors, genres, and tags—that are combined into a weighted "content soup" for TF-IDF representation. For the cultural component, the essential metadata fields are the *country code* and the *language code*, which jointly enable the mapping of each book onto a Hofstede cultural vector. It is a noteworthy property of the dataset that these two codes carry complementary reliability: the country code is near-complete (missing in approximately 0.02% of records), whereas the language code is missing in a substantial proportion of records (44.9%), a distribution that directly informs the fallback strategy described in Section 3.5.1.

**Component 3 — Cultural Dimension Data.** The culturally aware component requires a numerical representation of national culture against which both books and users can be positioned. This requirement is satisfied by Hofstede's six-dimensional model, which assigns each of 119 countries a vector of six scores—Power Distance Index (PDI), Individualism vs. Collectivism (IDV), Masculinity vs. Femininity (MAS), Uncertainty Avoidance Index (UAI), Long-Term Orientation (LTO), and Indulgence vs. Restraint (IVR)—each measured on a 0–100 scale. This dataset is described in detail in Section 3.3.3.

The logical separation of these three data components is methodologically significant: it is precisely because collaborative filtering requires interaction data (Component 1) that it fails for users and books lacking such data, while the culturally aware model, which additionally leverages cultural dimension data (Component 3) linked through book metadata (Component 2), is able to generate predictions even in the absence of interaction history. This asymmetry is the empirical foundation of the cold-start advantage that the study seeks to demonstrate.

### 3.3.2 PRIMARY DATASET

The primary dataset is the **UCSD Book Graph**, a large-scale corpus derived from the Goodreads platform and released by Wan and McAuley (2018). Goodreads is the largest social cataloguing website for books, and the corpus extracted from it represents one of the most extensive publicly available resources for research on literary recommendation. The corpus is distributed as a collection of JSON-lines and CSV files, the most relevant of which are summarised in Table 3.1.

| File | Format | Size | Records | Description |
|:-----|:-------|:-----|:--------|:------------|
| `goodreads_books.json` | JSON Lines | 9.20 GB | 2,360,655 | Book metadata: title, description, authors, country code, language code, ratings, pages, publication year, shelves |
| `goodreads_interactions.csv` | CSV | 4.21 GB | 222,824,625 | User–item interactions: user_id, book_id, read status, rating, review flag |
| `goodreads_book_authors.json` | JSON Lines | 105.9 MB | 829,529 | Author identifiers and names |
| `goodreads_book_genres_initial.json` | JSON Lines | 199.9 MB | 2,360,655 | Genre labels derived from user shelving |
| `goodreads_book_series.json` | JSON Lines | 111.1 MB | 400,390 | Series grouping metadata |
| `book_id_map.csv` | CSV | 37.8 MB | 2,360,650 | Mapping between book ID spaces |
| `user_id_map.csv` | CSV | 34.9 MB | 876,145 | Mapping between user ID spaces |

> **[IMAGE PLACEHOLDER — Table 3.1]**
> *Summary of the UCSD Book Graph files used in this study. Format this as a table in your document; columns are File, Format, Size, Records, and Description, with the seven rows shown above.*

**Book Metadata.** The `goodreads_books.json` file is the richest source of descriptive metadata. Each line is a JSON object containing, among other fields, the book's title, textual description, list of contributing authors (with author identifiers), the country of publication encoded as a two-letter code, the language of the edition encoded as a language code, the number of pages, the publication year, the average rating, the total number of ratings, and the "popular shelves"—a list of user-generated categorical labels. These fields are selectively consumed: the content-based model uses the title, description, authors, genres, and shelves, while the cultural model uses only the country and language codes.

**Interaction Data.** The `goodreads_interactions.csv` file contains one row per user–book interaction. Each interaction records the user identifier, the book identifier, a binary read flag, an explicit rating on a one-to-five scale (with zero denoting the absence of a rating), and a binary flag indicating whether the user also authored a review. An important characteristic of this dataset is that only a minority of interactions carry an explicit rating: of the approximately 222.8 million interactions, roughly 101.2 million include a non-zero rating, while the remaining 121.7 million (54.6%) represent unrated engagements such as "to-read" shelvings. This distinction matters because the recommender models are trained exclusively on the rated subset, while the unrated interactions contribute to the measurement of data sparsity. The rating distribution itself is heavily left-skewed—with a mode of four stars—a self-selection bias characteristic of user-generated ratings, in which users disproportionately rate books they have enjoyed; this skew is accounted for in the model training, which must differentiate quality within a narrow band of predominantly high ratings.

**Scale and Sparsity.** The corpus is notable both for its scale and for its extreme sparsity. Across 2.36 million books and 876,000 users, only approximately 101 million ratings are observed, yielding a user–item interaction matrix sparsity of approximately 99.9935%. Equivalently, the median book receives only four ratings, and more than half of all books receive four or fewer. This extreme sparsity is not a peripheral detail but a central motivating fact of the study: it is precisely this long-tail distribution—in which a small number of bestsellers command the overwhelming majority of attention while the vast majority of books remain virtually invisible to collaborative filtering—that the culturally aware model is designed to counteract. The consequences of this distribution for model design are developed throughout this chapter.

**Catalogue Subsampling.** Although the raw corpus is very large, the memory and computational constraints of the development environment necessitated a bounded catalogue for model training and evaluation. The implementation therefore streams the book metadata files and retains a bounded subset—50,000 books for the deployed system and 15,000 for the formal evaluation harness—filtered to those records that possess a title, a valid identifier, and a description of at least twenty characters. Interactions are likewise filtered to those referencing books within the retained catalogue. This subsampling strategy is a deliberate trade-off between coverage and tractability: it preserves the structural properties of the long tail (the retained sample still exhibits a median of four ratings per book) while reducing the corpus to a size that can be processed in memory on a single machine.

### 3.3.3 CULTURAL DIMENSIONS DATA SOURCE

The cultural dimension data is derived from Geert Hofstede's Cultural Dimensions Theory, the most widely adopted quantitative framework for the comparative analysis of national cultures (Hofstede, 2011). The framework operationalises culture along six orthogonal axes, each scored on a continuous 0–100 scale:

1. **Power Distance Index (PDI):** the degree to which a society accepts hierarchical and unequal distributions of power.
2. **Individualism vs. Collectivism (IDV):** the societal preference for individual autonomy versus tightly knit communal frameworks.
3. **Masculinity vs. Femininity (MAS):** the cultural emphasis on competition, achievement, and material reward versus cooperation and quality of life.
4. **Uncertainty Avoidance Index (UAI):** the extent to which a society tolerates ambiguity and relies on strict behavioural codes.
5. **Long-Term vs. Short-Term Orientation (LTO):** the preference for pragmatic, future-oriented adaptation versus normative adherence to tradition.
6. **Indulgence vs. Restraint (IVR):** the degree to which a culture permits the free gratification of human desires versus strict social regulation.

In this study, the six scores for a given country are assembled into an ordered vector $C = [\mathrm{PDI}, \mathrm{IDV}, \mathrm{MAS}, \mathrm{UAI}, \mathrm{LTO}, \mathrm{IVR}] \in \mathbb{R}^6$, which serves as the quantitative fingerprint of that country's culture. This vector is the atomic unit of the cultural feature-engineering process described in Section 3.5.

**Dataset Coverage and Missing Values.** The Hofstede dataset employed in this study, stored as `hofstede.csv`, contains profiles for 119 countries, spanning all six inhabited continents. The six dimensions exhibit differing degrees of completeness. PDI, IDV, MAS, and UAI are complete across all 119 countries. However, LTO is missing for 10 countries, and IVR is missing for 20 countries—a reflection of the fact that these two dimensions were added to the framework later than the original four and have not been measured for the full set of nations. These missing values are handled through median imputation during preprocessing (Section 3.4.4), a procedure that substitutes the column median for any absent score and thereby preserves a complete six-dimensional vector for every country.

**Relevance to African Literature.** The choice of Hofstede's framework is particularly apt for the African context that motivates this research. The framework's dimensions capture precisely the cultural contrasts that distinguish African literary traditions from their Western counterparts and that conventional recommender systems fail to model. Most notably, the Individualism–Collectivism axis sharply differentiates the collectivistic cultures of many West and East African nations—which score in the low twenties, reflecting a strong orientation towards community, kinship, and social consensus—from the highly individualistic cultures of Western nations such as the United States, which scores 91. Similarly, the Power Distance axis reflects the greater tolerance of hierarchical social structures characteristic of many African societies. These are not incidental attributes but core determinants of the narrative structures, thematic preoccupations, and social functions of African literature; a recommendation system that encodes them is therefore equipped to reason about literary relevance in a manner that culturally agnostic systems cannot. This line of argument, developed in the literature review, is here given computational form through the feature-engineering procedures of Section 3.5.

---

## 3.4 DATA PREPROCESSING

The raw corpus described in Section 3.3 is not directly amenable to machine learning. It is large, distributed across multiple files in inconsistent formats, riddled with missing values, and characterised by a profusion of noisy textual fields. This section describes the preprocessing procedures by which the raw data is transformed into clean, structured, analysis-ready tables. The procedures are organised into four sub-stages: streaming-based parsing, identifier alignment, text cleaning, and missing-value handling. Each sub-stage is described with reference to its rationale and its implementation in the project's codebase.

### 3.4.1 STREAMING PARSING OF LARGE FILES

The first preprocessing challenge is one of scale. The primary data files—`goodreads_books.json` at 9.20 GB and `goodreads_interactions.csv` at 4.21 GB—are too large to be loaded into memory in their entirety, particularly on the commodity hardware used for development. The preprocessing therefore adopts a **streaming** strategy, in which files are read incrementally rather than all at once.

For the JSON-lines files, this is achieved by iterating line-by-line: each line corresponds to a single JSON object, which is parsed individually and either retained or discarded according to the filtering criteria. This approach bounds memory usage to the size of a single record rather than the size of the file. For the CSV interaction file, the Pandas library's chunked reading facility is employed, in which the file is consumed in fixed-size blocks (`chunksize=100000` rows), each block is filtered, and the surviving rows are accumulated. This permits the interaction data—whose raw form exceeds 222 million rows—to be processed on a single machine without exhausting available memory.

An additional consequence of the streaming approach is that it enables **early termination**. Because the study operates on a bounded catalogue (Section 3.3.2), the interaction loader need not scan the entire file: it accumulates ratings only until a predetermined cap is reached (30,000 for the evaluation harness and 50,000 for the deployed system), at which point the scan terminates. This property is exploited to keep the preprocessing stage computationally tractable without sacrificing the representativeness of the retained sample.

### 3.4.2 IDENTIFIER ALIGNMENT AND CATALOGUE FILTERING

The second challenge is the alignment of identifiers across files. The corpus employs multiple, non-identical identifier spaces: the JSON metadata files reference books by one identifier scheme, while the interaction CSV references them by another. A dedicated mapping file, `book_id_map.csv`, provides the correspondence between the two spaces, and an analogous `user_id_map.csv` provides the correspondence for users. The preprocessing stage loads these mapping files and uses them to reconcile identifiers, ensuring that a book referenced in the interactions can be located in the metadata and vice versa.

Alignment proceeds as follows. First, the book metadata is streamed and each retained book is annotated with its mapped CSV identifier. Second, a set of valid CSV identifiers is assembled from the retained books. Third, the interaction stream is filtered to retain only those rows whose book identifier belongs to this valid set. This two-way filtering guarantees that every book appearing in the training data possesses full metadata, and that no orphaned interactions—references to books outside the retained catalogue—contaminate the dataset. The book identifiers are then re-indexed into a contiguous integer space (`book_idx`), and the user identifiers likewise (`user_idx`), so that the models operate on dense, zero-based indices rather than the sparse, arbitrary identifiers of the raw corpus. This re-indexing is a prerequisite for the one-hot feature encoding employed by the Factorization Machines (Section 3.6.3).

The catalogue filtering itself applies three acceptance criteria to each candidate book: it must possess a non-empty title, a valid identifier, and a description of at least twenty characters. These criteria serve a dual purpose. The title and identifier requirements ensure the book can be meaningfully presented to a user and unambiguously referenced; the description-length requirement ensures the book is a substantive literary record rather than an empty placeholder, and guarantees that the content-based baseline has textual material upon which to operate.

### 3.4.3 TEXT CLEANING AND CONTENT REPRESENTATION

The textual fields of the book metadata—particularly the description—are extracted from web pages and consequently contain HTML markup and other formatting artefacts. A cleaning procedure strips HTML tags from the description field using a regular-expression substitution, reducing a raw description such as `<p>A <b>novel</b> of ...</p>` to its plain-text equivalent. This cleaning is necessary for two reasons: first, because the content-based model treats the description as raw text for TF-IDF vectorisation, and the presence of markup would introduce spurious "words" such as tag names into the feature space; and second, because the description is surfaced to end users in the deployed interface, where markup would degrade readability.

A second form of cleaning concerns the **genre and shelf labels**. The genre labels are derived from Goodreads user shelving and consequently include a large number of non-informative tags—such as `to-read`, `currently-reading`, `owned`, `favorites`, and various year markers (`read-in-2019`)—that describe a user's personal organisational scheme rather than the content of the book. A curated denylist of such noisy tags is applied to filter the shelf data, so that only meaningful, content-descriptive labels are retained. The genres, together with the cleaned description, title, and author names, are subsequently combined into a weighted "content soup" for the content-based baseline (Section 3.6.1).

### 3.4.4 MISSING VALUE HANDLING

The corpus exhibits missing values in several fields, each of which requires a tailored strategy.

**Language and Country Codes.** As established in Section 3.3.2, the language code is missing for approximately 44.9% of books, whereas the country code is missing for only 0.02%. Because the cultural mapping depends on these codes, their missingness has a direct bearing on the reliability of the cultural features. The strategy adopted is a **fallback chain** (fully specified in Section 3.5.1): the country code is consulted first, and only if it is unavailable or unmappable does the system fall back to the language code, and only if both are unavailable does it default to a global average cultural vector. This design choice reflects the empirical finding that the country code is the more reliable field, and it ensures that the cultural mapping degrades gracefully rather than failing outright.

**Hofstede Dimensions.** Within the cultural dimensions dataset, the LTO and IVR dimensions contain missing values for a minority of countries (10 and 20, respectively). These are handled by **median imputation**: each missing value is replaced with the median of the non-missing values for that dimension across all countries. Median imputation is preferred over mean imputation because it is robust to outliers, and over deletion because deletion would sacrifice entire countries—and with them, the cultural diversity of the mapping—for the sake of a single absent value. The imputation is performed once, at load time, and the resulting complete six-dimensional vectors are cached for reuse throughout training and prediction.

**Other Metadata.** Other fields with material missingness—such as the number of pages (32.4% missing) and the publication year (25.4% missing)—are not imputed, because these fields are not consumed by any model in the study's final pipeline. Their missingness is therefore acknowledged but does not require a corrective procedure, since these fields are not inputs to any model.

---

## 3.5 CULTURAL FEATURE ENGINEERING

This section describes the methodological heart of the research: the procedures by which abstract sociological theory is rendered computationally tractable. The overarching goal is to construct, for every (user, book) pair, a numerical feature representation that encodes not merely *what* the user and the book are—as in traditional collaborative or content-based systems—but *how closely their cultures align*. The section is organised into four sub-stages, corresponding to the four transformations applied to the raw data: the mapping of geographic and linguistic metadata onto Hofstede vectors, the bottom-up propagation of book-level vectors into user-level cultural profiles, the derivation of a twenty-feature alignment representation, and the normalisation of those features into a bounded range.

### 3.5.1 COUNTRY → HOFSTEDE VECTOR MAPPING

The first transformation assigns to each book a six-dimensional cultural vector $C_b \in \mathbb{R}^6$, representing the national culture associated with that book's country of publication. The mapping is accomplished through a three-tier fallback chain, ordered by the reliability of the available metadata.

**Tier 1 — Country Code.** The book's two-letter country code (for example, `us`, `gb`, `ng`, `za`) is first converted to its full country name through a fixed lookup table, and the full name is then matched against the Hofstede dataset's country index. If a match is found, the corresponding six-dimensional vector is returned. This tier is the most reliable, because the country code is near-complete in the corpus (0.02% missing) and unambiguously identifies a single nation.

**Tier 2 — Language Code Fallback.** If the country code is absent or does not map to a country in the Hofstede dataset, the book's language code is consulted as a proxy. Because language is correlated with culture—though imperfectly so, as discussed below—a heuristic mapping associates each of several major language codes with a representative national culture: Spanish to Spain, German to Germany, French to France, Japanese to Japan, Russian to Russia, Chinese to China, and Portuguese to Brazil. This tier is necessary precisely because the language code is far more frequently missing (44.9%) than the country code; it exists to salvage cultural information for the minority of books that lack a country code but retain a language code.

**Tier 3 — Global Average.** If neither the country code nor the language code yields a mapping, the book is assigned the **global average cultural vector**, computed as the element-wise mean of all 119 national vectors in the Hofstede dataset. This default represents an epistemically honest fallback: it encodes the position that, in the absence of any cultural signal, the safest assumption is the centroid of the world's cultures rather than any single culture's profile.

The ordering of these tiers is a deliberate design decision grounded in the empirical reliability of each metadata field. Because the country code is both more complete and more semantically precise than the language code, it is consulted first. The language-code tier, by contrast, embodies an acknowledged approximation: it assumes, for example, that a Portuguese-language book originates from Brazil rather than Portugal, Angola, or Mozambique—an assumption that is frequently correct given the demographic weight of Brazil among Lusophone publications, but which is nonetheless a source of cultural approximation. This limitation is a consequence of the metadata available in the source corpus and is noted explicitly in the study's limitations.

### 3.5.2 BOTTOM-UP USER CULTURAL PROFILE PROPAGATION

The second transformation derives a cultural profile for each *user*, denoted $C_u \in \mathbb{R}^6$. Unlike books—whose cultural affiliation can be read directly from metadata—users do not carry a country label in the interaction data, and even if they did, a user's cultural identity is more faithfully revealed by their reading behaviour than by their nominal geography. The study therefore adopts a **bottom-up propagation** strategy: a user's cultural vector is inferred from the cultural vectors of the books they have rated highly.

Formally, let $I_u^{+}$ denote the set of books that user $u$ has rated at or above a threshold of 3.0 stars—the rationale being that a high rating signals genuine affinity, whereas a low rating signals rejection and should not contribute positively to the profile. The user's cultural vector is then defined as the arithmetic mean of the cultural vectors of their highly-rated books:

$$C_u = \frac{1}{|I_u^{+}|} \sum_{b \in I_u^{+}} C_b$$

Two edge cases require explicit handling. First, if a user has rated *no* book at or above 3.0 stars, the threshold criterion is relaxed and the profile is computed over the user's entire rating history, on the grounds that some cultural signal—even from mixed ratings—is preferable to none. Second, if a user has no rating history at all (the pure cold-start case), the profile defaults to the global average cultural vector.

The bottom-up strategy is significant for three reasons. First, it is **data-driven**: the profile is not imposed by the researcher but emerges organically from the user's demonstrated preferences, thereby avoiding the essentialism of assuming that a user's culture is determined solely by their nationality. Second, it is **dynamic**: because the profile is a function of the rating history, it can be recomputed incrementally as the user rates new books—a property that is exploited in the deployed system to provide real-time recalibration of recommendations (Section 3.6.5). Third, it is **theory-consistent**: the procedure operationalises the premise, argued in the literature review, that in collectivistic cultures a reader's affinity for a book reflects a cultural consensus rather than purely idiosyncratic taste, and that this consensus is best captured by aggregating over the books the reader has endorsed.

### 3.5.3 CULTURAL ALIGNMENT FEATURE SET

The third transformation constructs the feature vector that is actually fed to the model. Given a user profile $C_u$ and a book vector $C_b$, the objective is to capture not merely the two vectors in isolation, but the *relationship* between them. The study derives a twenty-feature representation, assembled from four complementary families of features:

**Family 1 — User Hofstede Scores (6 features).** The six components of the user's normalised cultural vector, $C_u / 100$, each scaled to the unit interval. These features allow the model to learn direct linear effects—for example, that users from high-Indulgence cultures systematically assign higher ratings overall, or that high-Uncertainty-Avoidance users display systematically more conservative rating behaviour.

**Family 2 — Book Hofstede Scores (6 features).** The six components of the book's normalised cultural vector, $C_b / 100$. These allow the model to learn item-level effects—for example, that books originating from particular cultural regions receive systematically different ratings.

**Family 3 — Aggregate Distance and Alignment (2 features).** Two scalar quantities summarising the overall cultural relationship:

(a) **Normalised Euclidean Cultural Distance**, defined as

$$d_E(C_u, C_b) = \frac{\lVert C_u - C_b \rVert_2}{100\sqrt{6}}$$

which is bounded in $[0, 1]$ because the numerator is the Euclidean norm of a difference of two vectors each confined to the $[0, 100]^6$ cube. A value of zero denotes identical cultural vectors; a value of one denotes maximal separation.

(b) **Cosine Cultural Similarity**, defined as

$$\cos_{sim}(C_u, C_b) = \frac{C_u \cdot C_b}{\lVert C_u \rVert_2 \, \lVert C_b \rVert_2}$$

which captures the *directional* alignment of the two vectors independent of their magnitude. Because the Hofstede vectors are non-negative, the cosine similarity is confined to $[0, 1]$, with a value of one denoting perfect directional alignment and a value of zero denoting orthogonality; it complements the Euclidean distance by measuring whether two cultures lean in the same direction even when their magnitudes differ.

The inclusion of *both* a magnitude-based distance and a direction-based similarity is deliberate. The two measures capture different, non-redundant information—a point that is illustrated by the empirical scatter plot presented in Chapter 4, which demonstrates that Euclidean distance and cosine similarity exhibit a smooth but distinctly non-linear relationship across country pairs. Feeding both to the model therefore provides complementary mathematical signals rather than duplicating one another.

**Family 4 — Dimension-Wise Absolute Gaps (6 features).** The absolute difference between the user's and the book's scores on each of the six dimensions, each normalised to $[0, 1]$:

$$\Delta_{\dim} = \frac{|C_u - C_b|}{100}$$

These per-dimension gaps allow the model to learn *which* cultural differences matter. A large gap in Individualism may exert a different influence on rating behaviour than a large gap in Uncertainty Avoidance, and by exposing each gap as a separate feature, the model is permitted to weight them differentially during training. The concatenation of the four families yields the final twenty-dimensional representation:

$$x_{\text{cultural}} = \left[\, \frac{C_u}{100}\ (6) \ \middle\| \ \frac{C_b}{100}\ (6) \ \middle\| \ d_E\ (1) \ \middle\| \ \cos_{sim}\ (1) \ \middle\| \ \Delta_{\dim}\ (6)\, \right]$$

This feature set constitutes the empirical claim at the centre of the research: that a compact, theory-grounded representation of cultural alignment, added to a Factorization Machine's feature space, is sufficient to measurably improve recommendation quality over culturally agnostic baselines. The validation of this claim is the subject of the evaluation in Section 3.8 and Chapter 4.

### 3.5.4 FEATURE STANDARDISATION

The final transformation concerns the scale of the features. The cultural features are constructed to be bounded: the six-dimensional score vectors are divided by 100, the Euclidean distance is normalised by its theoretical maximum ($100\sqrt{6}$), and the absolute gaps are likewise divided by 100, so that every component of the twenty-feature vector lies in the unit interval. This normalisation is functionally equivalent to a bounded min–max scaling and serves a critical purpose in the context of Factorization Machine training.

The rationale is that the Factorization Machine's stochastic gradient descent update rule scales its gradients by the magnitude of the feature values (Section 3.6.3). If features are permitted to assume wildly different magnitudes—say, Hofstede scores in the range $[0, 100]$ alongside one-hot indicators in $\{0, 1\}$—then the high-magnitude features would dominate the gradient updates and bias the learning process. By normalising all cultural features to $[0, 1]$, the study ensures that they contribute to the model on an equal footing with the binary identity features, preventing any single feature family from monopolising the optimisation. This standardisation is applied identically at training time and at prediction time, guaranteeing consistency between the two phases.

---

## 3.6 MODEL DEVELOPMENT

This section describes the family of predictive models constructed for the study and the mathematical formulations that govern them. The research develops a total of six models, arranged in three tiers of increasing sophistication. The first tier comprises two *baseline* models—a content-based recommender and a collaborative-filtering model—which represent the culturally agnostic techniques against which the proposed approach is benchmarked. The second tier comprises three variants of a *culturally aware Factorization Machine*, which progressively integrate cultural information into the model's feature space. The third tier comprises a single *hybrid engine*, which unifies the strongest culturally aware model with the collaborative-filtering baseline through an adaptive switching mechanism. This comparative family is essential to the study's validity, because the research claim is not merely that the proposed model performs well, but that its performance improvement is attributable specifically to the integration of cultural features.

### 3.6.1 CONTENT-BASED BASELINE

The content-based baseline is the simplest model in the family and serves as the lower bound against which the more sophisticated techniques are judged. It is a purely text-driven model that makes no use of collaborative signal or cultural theory; its predictions are derived solely from the similarity between a candidate book's textual description and the descriptions of books the user has previously rated.

The baseline operates in two stages. First, a **content soup** is constructed for each book by concatenating its title, author names, genre labels, and shelf tags, each with an appropriate repetition weight, together with its full textual description. The repetition of certain fields—titles repeated twice, authors thrice, genres and tags twice—encodes the prior that these categorical fields are more semantically informative, per unit of text, than the comparatively diffuse prose of the description. Second, a **TF-IDF vectoriser** is fitted over the corpus of content soups, transforming each book into a sparse vector in a high-dimensional term space (capped at 25,000 features). The similarity between two books is then measured by the cosine of the angle between their TF-IDF vectors.

Given a user's rating history, the predicted rating for a candidate book is computed as a **similarity-weighted average** of the user's historical ratings. Formally, let $H_u = \{(b_j, r_j)\}$ denote the set of books the user has rated, with $r_j$ the corresponding rating, and let $\text{sim}(b, b_j)$ denote the cosine similarity between the candidate book $b$ and a historically rated book $b_j$. The predicted rating is:

$$\hat{y}_{\text{CB}}(u, b) = \frac{\sum_{(b_j, r_j) \in H_u} \text{sim}(b, b_j) \cdot r_j}{\sum_{(b_j, r_j) \in H_u} \text{sim}(b, b_j)}$$

with similarities clamped to be non-negative, so that dissimilar books cannot exert a negative influence. When a user has no history (the cold-start case), the model defaults to a global mean rating of 3.5. The content-based baseline is deliberately simple: it requires no training beyond the one-time fitting of the TF-IDF vectoriser, and it serves as a concrete instantiation of the content-based filtering paradigm described in the literature review. Its known weaknesses—an inability to recommend across thematic boundaries and a tendency toward narrow, self-similar recommendations—are precisely the weaknesses that the collaborative and cultural models are designed to overcome.

### 3.6.2 COLLABORATIVE FILTERING BASELINE (SVD++)

The second baseline is a **latent factor model** drawn from the collaborative-filtering paradigm. Whereas the content-based model represents books by their textual content, the collaborative model represents both users and books by latent vectors inferred purely from the pattern of observed ratings, on the principle that users who agreed in the past will agree in the future.

The study employs the **SVD++** algorithm, an extension of matrix factorisation that augments the standard latent-factor formulation with a term capturing *implicit feedback*. SVD++ is chosen because it is among the strongest classical collaborative-filtering techniques, and therefore constitutes a rigorous benchmark: if the culturally aware model can match or exceed SVD++, the result is meaningful. The predicted rating of user $u$ for book $i$ is modelled as:

$$\hat{y}_{\text{SVD++}}(u, i) = \mu + b_u + b_i + q_i^T \left( p_u + |I_u|^{-1/2} \sum_{j \in I_u} y_j \right)$$

where $\mu$ is the global mean rating; $b_u$ and $b_i$ are user- and item-specific bias terms capturing systematic deviations from the mean; $q_i$ is the latent factor vector of book $i$; $p_u$ is the latent factor vector of user $u$; and the final summation aggregates the latent implicit-feedback vectors $y_j$ over the set $I_u$ of items with which the user has interacted, regardless of the rating value assigned. The normalisation by $|I_u|^{-1/2}$ prevents the implicit term from growing unboundedly with the length of a user's history.

The model is trained by stochastic gradient descent to minimise a regularised squared error between predicted and observed ratings, with latent dimensionality $k = 10$ and eight training epochs in the study's configuration. The implementation is provided by the `scikit-surprise` library, wrapped in a thin interface to conform to the study's unified model API.

The defining characteristic of SVD++—and the source of both its strength and its weakness—is its complete dependence on historical interaction data. For a user with a rich rating history, the latent factors $p_u$ are well-identified and the model achieves strong predictive accuracy. For a user with no history, however, $p_u$ is undefined, and the model collapses to a global prior that cannot be personalised. This cold-start fragility is the precise failure mode that the culturally aware models of Section 3.6.4 are designed to address.

### 3.6.3 FACTORIZATION MACHINE FORMULATION

The culturally aware models are built upon the **Factorization Machine** (FM), a general-purpose supervised predictor introduced by Rendle (2010) and chosen here for its capacity to model interactions between heterogeneous feature types—in particular, between categorical identity features and continuous cultural features—within a single unified framework. The FM generalises both linear regression and matrix factorisation: it models all single-variable effects and all pairwise variable interactions, with the interaction weights factorised so that the model remains estimable even when most variable pairs never co-occur in the training data.

Formally, let $x \in \mathbb{R}^d$ denote the input feature vector for a single training instance (a user–book pair). The FM prediction is defined as:

$$\hat{y}(x) = w_0 + \sum_{i=1}^{d} w_i x_i + \sum_{i=1}^{d} \sum_{j=i+1}^{d} \langle v_i, v_j \rangle\, x_i x_j$$

where $w_0 \in \mathbb{R}$ is the global bias; $w_i \in \mathbb{R}$ is the first-order weight of the $i$-th feature; and $\langle v_i, v_j \rangle$ denotes the inner product of the $k$-dimensional latent factor vectors $v_i, v_j \in \mathbb{R}^k$ associated with features $i$ and $j$. The first summation models linear feature effects; the double summation models pairwise interactions.

The computational cost of the pairwise sum is nominally $O(k\, d^2)$, which would be prohibitive for large feature spaces. The FM exploits the algebraic identity:

$$\sum_{i=1}^{d} \sum_{j=i+1}^{d} \langle v_i, v_j \rangle\, x_i x_j = \frac{1}{2} \sum_{f=1}^{k} \left[ \left( \sum_{i=1}^{d} v_{i,f} x_i \right)^2 - \sum_{i=1}^{d} v_{i,f}^2 x_i^2 \right]$$

which reduces the computation to linear time $O(k\, d)$. This reformulation is critical to the study's feasibility: it permits the model to operate over a feature space whose dimensionality runs into the tens of thousands (comprising one-hot user and book identities plus the twenty cultural features), within a single forward pass per training sample.

The model is trained by **stochastic gradient descent**, minimising the squared error between the predicted rating and the observed rating, with an L2 regularisation term to prevent overfitting. The prediction is clipped to the valid rating range $[1, 5]$ after each update. The factorisation rank $k$, the learning rate, the regularisation strength, and the number of epochs are hyperparameters tuned empirically; the study's configuration uses $k = 10$, a learning rate of 0.01, a regularisation strength of 0.03, and eight epochs.

The FM is the natural vehicle for the cultural feature engineering of Section 3.5 precisely because its pairwise-interaction term is capable of learning the *joint* effects of cultural alignment—for example, that a high-Power-Distance user and a high-Power-Distance book jointly signal an elevated rating—that a purely linear model would miss. The three variants described next differ in the composition of the feature vector $x$ that they receive.

### 3.6.4 CULTURALLY AWARE FACTORIZATION MACHINES (FM v1, FM v2, FM v3)

The three culturally aware models share the same FM architecture of Section 3.6.3 and differ only in the richness of their feature vectors, in a deliberate progression from a minimal baseline to the fully specified model.

**FM v1 — Baseline Hofstede Integration.** The first variant represents the minimal integration of cultural information. Its feature vector concatenates three components: a one-hot encoding of the user identity, a one-hot encoding of the book identity, and the raw Hofstede scores of the user and the book—six normalised components each, totalling twelve continuous features. Formally, for a user $u$ and book $b$:

$$x^{(v1)} = \left[\, \text{user\_id}(u) \ \middle\| \ \text{book\_id}(b) \ \middle\| \ \frac{C_u}{100} \ \middle\| \ \frac{C_b}{100}\, \right]$$

FM v1 is able to learn first-order cultural effects and pairwise interactions *between* user identity, book identity, and the individual Hofstede components, but it does not encode any explicit *relational* quantity linking user and book. It therefore serves as a controlled baseline for assessing whether the explicit distance and alignment features of FM v2 contribute genuine predictive value.

**FM v2 — Explicit Cultural Distance and Alignment (the selected model).** The second variant augments FM v1's feature vector by replacing the twelve raw Hofstede scores with the full twenty-dimensional cultural alignment representation developed in Section 3.5.3:

$$x^{(v2)} = \left[\, \text{user\_id}(u) \ \middle\| \ \text{book\_id}(b) \ \middle\| \ x_{\text{cultural}} \, \right]$$

where $x_{\text{cultural}}$ is the twenty-feature vector comprising the six normalised user scores, the six normalised book scores, the normalised Euclidean distance, the cosine similarity, and the six dimension-wise absolute gaps. The distinguishing property of FM v2 is that it supplies the model with *relational* features—the distance and alignment between the user's and the book's cultures—that FM v1 lacks. It is these relational features that enable the model to reason about cultural proximity directly, and it is FM v2 that is carried forward as the selected model for the hybrid engine and the deployed system. The empirical demonstration that FM v2 outperforms FM v1—with a high degree of statistical significance—constitutes the study's primary evidence that explicit cultural distance modelling yields genuine predictive benefit; this result is reported in Chapter 4.

**FM v3 — Genre-Augmented Variant.** The third variant explores the orthogonal question of whether the cultural signal can be enriched with categorical *content* information. FM v3 extends FM v2's feature vector with a **multi-hot genre encoding**: a sparse vector in which the dimensions correspond to the study's genre vocabulary (restricted to genres occurring at least five times), and in which a book activates the dimensions of its genres with values normalised by $1/\sqrt{|G|}$, where $|G|$ is the number of genres the book carries. This normalisation ensures that books with many genres do not dominate the feature space. Formally:

$$x^{(v3)} = \left[\, \text{user\_id}(u) \ \middle\| \ \text{book\_id}(b) \ \middle\| \ \text{genres}(b) \ \middle\| \ x_{\text{cultural}} \, \right]$$

FM v3 permits the model to learn interactions between cultural alignment and genre—for example, that a culturally proximal book in a particular genre is rated differently from a culturally distal book in the same genre. It is evaluated as a secondary contribution; the primary model remains FM v2, on the grounds that FM v3's additional complexity must justify itself against the simpler and more parsimonious FM v2.

### 3.6.5 SWITCHING-WEIGHTED HYBRID ENGINE

The final model in the family is not a new predictor but a **composition** of the two strongest models identified in the preceding sections—the culturally aware FM v2 and the collaborative SVD++—designed to exploit their complementary strengths. The empirical analysis reported in Chapter 4 establishes that FM v2 dominates SVD++ in the warm-start regime (approximately 1–10 ratings), where collaborative signal is sparse but cultural signal is available, whereas SVD++ regains the advantage for mature users with more than ten ratings, where latent factors are well-identified. This complementary behaviour motivates a hybrid that delegates to each model in the regime where it excels.

The hybrid adopts a **switching-weighted** strategy, governed by a single threshold parameter $T$ and a single blend weight $\alpha$. Let $n_u$ denote the number of ratings available for user $u$ in the training set. The hybrid prediction is defined as:

$$\hat{y}_{\text{hybrid}}(u, b) = \begin{cases} \hat{y}_{\text{FM v2}}(u, b) & \text{if } n_u < T \\[4pt] \alpha \, \hat{y}_{\text{FM v2}}(u, b) + (1 - \alpha)\, \hat{y}_{\text{SVD++}}(u, b) & \text{if } n_u \geq T \end{cases}$$

In the cold- and warm-start regime ($n_u < T$), the hybrid delegates entirely to FM v2, because SVD++ lacks the collaborative signal necessary for reliable prediction and its inclusion would degrade accuracy. In the mature-user regime ($n_u \geq T$), the hybrid blends the two predictions with weight $\alpha$ assigned to FM v2 and $(1 - \alpha)$ to SVD++.

The two hyperparameters are determined empirically by grid search over a dedicated validation partition, as described in Section 3.8.2. The grid search explores thresholds $T \in \{1, 3, 5, 8, 10\}$ and blend weights $\alpha \in \{0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9\}$, selecting the combination that minimises validation error. The optimal configuration identified by this procedure—and used throughout the deployed system—is $T = 1$ and $\alpha = 0.80$. The value $T = 1$ indicates that the switch to blended prediction occurs immediately upon the user's first rating, while the weight $\alpha = 0.80$ indicates that, even for mature users, the culturally aware model is assigned the dominant share of the prediction, with SVD++ contributing a corrective collaborative signal. This asymmetry is itself a meaningful empirical finding: it demonstrates that cultural awareness remains the more informative signal even when collaborative data is available, a result consistent with the study's central thesis.

The hybrid engine additionally supports **real-time profile recalibration**. When a user submits a new rating through the deployed interface, the engine appends the interaction to the user's history and recomputes their cultural profile $C_u$ as the average of the cultural vectors of their newly-eligible highly-rated books (Section 3.5.2). Because the FM v2 prediction is a function of $C_u$, this recomputation immediately updates the cultural component of the hybrid's output, allowing recommendations to reflect the user's evolving profile without requiring full model retraining. The SVD++ component, by contrast, remains fixed from its offline training, an approximation justified by the prohibitive cost of retraining a collaborative model on every interaction and deemed acceptable for the purposes of the deployed demonstration. This asymmetric update scheme is the operational realisation of the hybrid's design philosophy: the culturally aware component is inherently adaptive to new input, while the collaborative component is treated as a relatively stable prior.

---

## 3.7 SYSTEM ARCHITECTURE

This section describes the software architecture into which the models of Section 3.6 are embedded, and through which the recommendation engine is made available to end users. The architecture is organised as a three-tier client–server system: a web frontend, a RESTful application-programming-interface (API) backend, and a model/application server that hosts the trained models and the book catalogue in memory. The design is guided by three principles: separation of concerns, so that the user interface, the API layer, and the machine-learning layer can evolve independently; in-memory model serving, so that recommendations can be generated with sub-50-millisecond latency; and cloud portability, so that the system can be containerised and deployed on commodity cloud infrastructure.

### 3.7.1 OVERALL SYSTEM ARCHITECTURE

The overall architecture is illustrated in Figure 3.2. The three tiers are as follows.

**Presentation Tier.** A single-page application (SPA) built with React and served as static assets. The frontend presents the user with an onboarding flow, a recommendation feed, a search interface, a rating widget, and a personal cultural-profile panel. It communicates with the backend exclusively over HTTP, exchanging JSON payloads. The choice of a decoupled SPA—rather than server-rendered pages—reflects the need for rich, real-time interactivity: the recommendation feed updates in place when a user submits a rating, without a full page reload.

**Application Tier.** A FastAPI backend exposing a set of REST endpoints (Section 3.7.2). The backend holds a single in-memory application state comprising the trained hybrid model, the book catalogue, the Hofstede country profiles, and the active user sessions. On startup, it loads the catalogue and fits the hybrid engine once; subsequent requests are served entirely from memory, yielding the low latency required for interactive recommendation.

**Model Tier.** The trained model artefacts—the FM v2 weight tensors, the SVD++ latent factors, and the derived cultural vectors—together with the book metadata catalogue. In the deployed system these artefacts are pre-trained offline and serialised, then loaded into the application tier at startup. In the development configuration, the models are fitted in-process at startup rather than deserialised, a distinction of deployment convenience rather than of architecture.

```
┌──────────────────────┐         ┌──────────────────────┐         ┌──────────────────────┐
│   PRESENTATION       │         │   APPLICATION        │         │   MODEL TIER         │
│   (React SPA)        │  HTTP   │   (FastAPI backend)  │         │   (Pre-trained       │
│                      │◀───────▶│                      │────────▶│    artefacts)        │
│  - Onboarding        │  JSON   │  - REST endpoints    │ in-mem  │                      │
│  - Recommendation    │         │  - Session state     │  load   │  - FM v2 weights     │
│  - Search / Rate     │         │  - Hybrid engine     │         │  - SVD++ factors     │
│  - Cultural radar    │         │  - Book catalogue    │         │  - Cultural vectors  │
└──────────────────────┘         └──────────────────────┘         └──────────────────────┘
```

> **[IMAGE PLACEHOLDER — Figure 3.2]**
> *Three-tier system architecture: React SPA (presentation) communicating over HTTP/JSON with a FastAPI backend (application), which loads pre-trained model artefacts (FM v2 weights, SVD++ factors, cultural vectors) and the book catalogue into memory. Recreate as a three-column diagram; the ASCII sketch above gives the exact components and interaction directions.*

### 3.7.2 API LAYER SPECIFICATION

The backend exposes a small, purpose-built REST API whose endpoints correspond directly to the user's interaction flow. The endpoints are summarised in Table 3.2.

| Endpoint | Method | Purpose |
|:---------|:-------|:--------|
| `/api/health` | GET | System health and catalogue statistics |
| `/api/countries` | GET | List of 119 Hofstede country profiles |
| `/api/onboard` | POST | Create a user session from a country selection; return cold-start recommendations |
| `/api/rate` | POST | Submit a rating; recompute the cultural profile; return refreshed recommendations |
| `/api/recommend` | GET | Retrieve the top-K hybrid recommendations for a user |
| `/api/search` | GET | Full-text search of the catalogue by title, author, or genre |
| `/api/profile/{user_id}` | GET | A user's rating history and current Hofstede profile |
| `/api/cover/{book_id}` | GET | Resolve a book's cover image URL |

> **[IMAGE PLACEHOLDER — Table 3.2]**
> *REST API endpoint specification. Format as a table in your document; columns are Endpoint, Method, and Purpose, with the eight rows shown above.*

Two endpoints merit particular attention because they embody the system's cultural-awareness mechanism. The **`/api/onboard`** endpoint accepts a user's country selection and maps it to a Hofstede vector; it then generates a cold-start recommendation list by invoking the hybrid engine with the user's cultural profile and *no* interaction history—a request that the hybrid routes entirely to FM v2 (Section 3.6.5). This is the point at which the system's ability to recommend without collaborative data is exercised in practice. The **`/api/rate`** endpoint, conversely, receives a rating, appends it to the user's history, recomputes their cultural profile through the bottom-up propagation of Section 3.5.2, and returns a refreshed recommendation list—demonstrating the real-time profile recalibration that distinguishes the culturally aware model from static collaborative baselines.

Each recommendation returned by the API is accompanied by a **cultural alignment score**, computed as the complement of the normalised Euclidean distance between the user's and the book's cultural vectors. This score is surfaced to the user as a percentage, providing a transparent, human-interpretable rationale for why a particular book was recommended—an explanation that is impossible for a latent-factor model such as SVD++, whose internal representations are not human-readable.

### 3.7.3 CLOUD DEPLOYMENT TOPOLOGY

The system is packaged for cloud deployment using containerisation, in accordance with the study's aim of delivering a deployable, cloud-ready platform. The deployment topology is illustrated in Figure 3.3.

The application is containerised with Docker, using a multi-stage build that compiles the React frontend into static assets in a first stage and packages those assets alongside the Python backend and model artefacts in a second stage. The resulting container exposes the FastAPI server on a single port. In the target cloud environment, the container is deployed on an AWS Elastic Compute Cloud (EC2) instance—sized at the `t3.medium` tier (2 vCPUs, 4 GB memory), which is sufficient to hold the model artefacts and the 50,000-book catalogue in memory—behind an Nginx reverse proxy that terminates transport-layer security and serves the static frontend. A Docker Compose specification coordinates the orchestration of the backend container and its supporting services.

```
┌──────────────┐      ┌────────────────┐      ┌─────────────────────┐
│   Client     │ HTTPS│  Nginx         │      │  Docker Container   │
│   Browser    │─────▶│  (TLS + static │─────▶│  (FastAPI + models) │
│              │      │   + reverse    │      │                     │
└──────────────┘      │   proxy)       │      └─────────────────────┘
                      └────────────────┘               │
                                                       │
                                            ┌──────────▼──────────┐
                                            │  AWS EC2 (t3.medium)│
                                            │  2 vCPU / 4 GB RAM  │
                                            └─────────────────────┘
```

> **[IMAGE PLACEHOLDER — Figure 3.3]**
> *Cloud deployment topology: browser → Nginx (TLS termination + reverse proxy) → Docker container (FastAPI + models) hosted on an AWS EC2 t3.medium instance. Recreate as a layered diagram; the ASCII sketch above gives the exact components and flow.*

Two aspects of the deployment topology are notable from a methodological standpoint. First, the architecture is **stateless with respect to the model**: all model artefacts are immutable at runtime, and the only dynamic state—user sessions and rating histories—resides in process memory. This design keeps the deployment simple (no external database is required for the demonstration) at the cost of losing session state on restart, a trade-off explicitly accepted for the study's scope. Second, the containment of the entire system within a single Docker image is a deliberate choice to maximise portability: the same image that runs in local development can be deployed to any container-capable cloud host without modification, a property that supports the study's claim of delivering a genuinely deployable—rather than merely prototypical—platform.

---

## 3.8 EVALUATION STRATEGY

This section describes the methodology by which the models of Section 3.6 are evaluated and compared. The evaluation strategy is designed to answer three distinct questions, each of which requires a different analytical apparatus. The first question is *how accurately* each model predicts ratings, answered through error metrics. The second is *how well* each model ranks relevant books, answered through ranking metrics. The third is *whether the observed differences are genuine*, answered through statistical significance testing. The strategy is further stratified to isolate the effect of the cultural features in the cold-start regime that motivates the study. The section concludes by formalising the benchmarking procedure against the traditional baselines.

### 3.8.1 EVALUATION METRICS

The models are assessed on nine metrics, spanning three complementary dimensions of recommendation quality: predictive accuracy, ranking quality, and beyond-accuracy diversity. All metrics are computed independently for the active-user and cold-start partitions, as described in Section 3.8.2.

**Predictive Accuracy.** Two error metrics quantify how closely each model's predicted rating matches the user's actual rating.

The **Root Mean Square Error (RMSE)** is defined as:

$$\text{RMSE} = \sqrt{ \frac{1}{N} \sum_{(u,b) \in T} \left( \hat{y}_{u,b} - y_{u,b} \right)^2 }$$

and the **Mean Absolute Error (MAE)** as:

$$\text{MAE} = \frac{1}{N} \sum_{(u,b) \in T} \left| \hat{y}_{u,b} - y_{u,b} \right|$$

where $T$ denotes the test set, $N = |T|$ its cardinality, $\hat{y}_{u,b}$ the predicted rating, and $y_{u,b}$ the ground-truth rating. RMSE and MAE are complementary: RMSE penalises large errors quadratically and is therefore sensitive to outliers, while MAE treats all errors linearly and is more robust. Reporting both provides a balanced account of predictive error.

**Ranking Quality.** Because the deployed system presents a *ranked list* of recommendations rather than a single prediction, the accuracy of the top of that list is as important as the accuracy of the underlying scores. Four ranking metrics are employed, all computed over the top-$k$ recommendations (with $k = 5$ in the study's configuration).

**Precision@k** measures the fraction of the recommended list that is genuinely relevant:

$$\text{Precision@}k = \frac{1}{|U_{\text{test}}|} \sum_{u \in U_{\text{test}}} \frac{|L_u(k) \cap R_u|}{k}$$

**Recall@k** measures the fraction of the user's relevant books that the list succeeds in surfacing:

$$\text{Recall@}k = \frac{1}{|U_{\text{test}}|} \sum_{u \in U_{\text{test}}} \frac{|L_u(k) \cap R_u|}{|R_u|}$$

where $L_u(k)$ is the set of the top-$k$ recommended books for user $u$, and $R_u$ is the set of books that are relevant to user $u$, defined operationally as those the user rated at or above 3.5 stars in the test partition. Precision and Recall are combined into the harmonic mean **F1@k**:

$$\text{F1@}k = \frac{1}{|U_{\text{test}}|} \sum_{u \in U_{\text{test}}} \frac{2 \cdot \text{Precision}_u(k) \cdot \text{Recall}_u(k)}{\text{Precision}_u(k) + \text{Recall}_u(k)}$$

Finally, the **Normalised Discounted Cumulative Gain (nDCG@k)** accounts for the *position* of relevant items within the list, rewarding models that place relevant books near the top. The discounted cumulative gain for a single user's list is:

$$\text{DCG@}k = \sum_{i=1}^{k} \frac{\mathbb{1}[\, L_u(i) \in R_u \,]}{\log_2(i+1)}$$

where $L_u(i)$ denotes the $i$-th item in the ranked list and $\mathbb{1}[\cdot]$ the indicator function. The ideal DCG, $\text{IDCG@}k$, is the DCG of a hypothetical perfect ranking in which every position up to $k$ (bounded by the number of relevant items) is occupied by a relevant book. The nDCG is the ratio:

$$\text{nDCG@}k = \frac{1}{|U_{\text{test}}|} \sum_{u \in U_{\text{test}}} \frac{\text{DCG}_u(k)}{\text{IDCG}_u(k)}$$

The inclusion of nDCG alongside precision and recall is methodologically important: a model that places relevant books at ranks 1 and 2 is superior to one that places them at ranks 4 and 5, and only nDCG captures this distinction.

**Beyond-Accuracy Diversity.** Three further metrics assess qualities that are orthogonal to accuracy and that are central to the study's aim of surfacing long-tail literature.

**Intra-List Diversity (ILD)** measures the dissimilarity of the books within a single recommendation list, computed as the average pairwise dissimilarity:

$$\text{ILD} = \frac{1}{|U_{\text{test}}|} \sum_{u \in U_{\text{test}}} \frac{2}{k(k-1)} \sum_{i=1}^{k} \sum_{j=i+1}^{k} \left( 1 - \cos(\mathbf{t}_{L_u(i)}, \mathbf{t}_{L_u(j)}) \right)$$

where $\cos(\mathbf{t}_a, \mathbf{t}_b)$ is the cosine similarity between the TF-IDF vectors of books $a$ and $b$. A higher ILD indicates a more diverse—less self-similar—list.

**Novelty** measures the extent to which a model recommends unpopular, long-tail items. For each recommended book, its popularity is defined by the number of ratings it received in the training set, $\text{pop}(b)$, and its self-information is $-\log_2\!\left(\text{pop}(b)/N_{\text{train}}\right)$, which is large for obscure items and small for bestsellers. Novelty is the mean self-information over all recommendations:

$$\text{Novelty} = \frac{1}{|U_{\text{test}}|} \sum_{u \in U_{\text{test}}} \frac{1}{k} \sum_{b \in L_u(k)} -\log_2\!\left( \frac{\text{pop}(b)}{N_{\text{train}}} \right)$$

**Catalog Coverage** measures the breadth of the catalogue that the model actually surfaces, defined as the fraction of candidate books that appear in at least one user's recommendation list. A model that recommends the same few bestsellers to every user achieves low coverage, whereas a model that reaches into the long tail achieves high coverage:

$$\text{Coverage} = \frac{\left| \bigcup_{u \in U_{\text{test}}} L_u(k) \right|}{|I_{\text{candidate}}|}$$

These three metrics are the study's operational answer to the popularity-bias problem identified in the literature review: a recommender that merely maximises accuracy will tend toward low novelty and low coverage, whereas the culturally aware model is hypothesised to trade a modest amount of accuracy for substantially greater diversity, novelty, and reach into the long tail. The results reported in Chapter 4 evaluate this hypothesis directly.

### 3.8.2 VALIDATION AND TESTING TECHNIQUES

The evaluation employs a **multi-split cross-validation** procedure that averages results over five independently seeded data partitions, mitigating the risk that any observed difference is an artefact of a particular split. The procedure is as follows.

**Cold-Start Holdout.** First, 15% of users are randomly selected to form the *cold-start* partition. These users' interactions are held out entirely from the training set and reserved for evaluating cold-start performance. For these users, the models are evaluated without access to any training history—the collaborative and content-based models are given empty histories, and the culturally aware models are given only the global average cultural vector. This partition directly operationalises the study's central scenario: a new reader, about whom the system knows nothing but whose culture it can infer.

**Train/Test Split.** For the remaining 85% of users, interactions are split 80/20 into training and test partitions by independent random assignment of each interaction, so that each active user contributes ratings to both partitions. The training partition is used to fit all models; the test partition is used to evaluate them.

**Replication.** The entire procedure is repeated for five distinct random seeds (42, 101, 777, 999, and 2024), each producing a fresh cold-start selection and train/test split. Every metric is computed separately for each seed, and the final result is reported as the mean ± standard deviation across the five runs. This replication is essential: recommendation datasets are noisy, and a single split could yield a misleadingly favourable or unfavourable comparison. The standard deviation additionally provides a measure of the stability of each model's performance.

**Stratified History Analysis.** To characterise how model performance evolves across a user's lifecycle, the test set is further stratified into four interaction-depth buckets—0 ratings (pure cold start), 1–3 ratings, 4–10 ratings, and more than 10 ratings—according to each user's training-set history length. The MAE is reported separately for each bucket, revealing the crossover point at which the culturally aware model's advantage over SVD++ diminishes as collaborative signal accumulates. This stratification is the analytical bridge between the cold-start evaluation and the hybrid design of Section 3.6.5: it is the empirical evidence that motivates the choice of the switching threshold.

**Hybrid Hyperparameter Tuning Split.** The tuning of the hybrid engine's two hyperparameters (Section 3.6.5) employs a separate, dedicated data partition, distinct from the five-seed evaluation described above. For this purpose, the interactions of non-cold-start users are divided three ways: 70% into a training partition, 15% into a validation partition, and 15% into a held-out test partition, with a further 15% of users held out for cold-start testing. The training partition is used to fit the constituent models; the validation partition is used to evaluate every candidate combination of the threshold $T$ and blend weight $\alpha$ in the grid search; and the held-out test partition is reserved for the final, unbiased comparison of the tuned hybrid against its standalone constituent models. This three-way split—with a validation partition explicitly reserved for hyperparameter selection—prevents the grid search from overfitting the evaluation data, and ensures that the hybrid's reported performance is measured on data it has never influenced. The results of this tuning and the resulting benchmark are reported in Chapter 4.

### 3.8.3 STATISTICAL SIGNIFICANCE TESTING

The reporting of point estimates alone is insufficient to establish that the culturally aware model genuinely outperforms its baselines, because a small apparent difference may be attributable to random variation. The evaluation therefore accompanies the aggregate metrics with paired statistical significance tests on the per-instance absolute errors.

Specifically, for each test instance, the absolute error $| \hat{y}_{u,b} - y_{u,b} |$ is recorded for every model, yielding paired observations across models for the identical set of test instances. Two complementary tests are then applied:

The **paired Student's t-test** tests the null hypothesis that the mean difference in absolute error between two models is zero. Because the errors are paired across identical instances, the paired test has substantially greater statistical power than an unpaired comparison, isolating the effect of the model from the effect of instance difficulty.

The **Wilcoxon signed-rank test** is applied as a non-parametric complement. Unlike the t-test, it does not assume that the differences are normally distributed—an assumption that is unlikely to hold for absolute errors, which are bounded below by zero and positively skewed. The Wilcoxon test therefore provides a robustness check against violations of normality.

The paired t-test is applied to the full set of test instances, while the Wilcoxon test is applied to a bounded subset of the instances for computational tractability. Both tests are conducted for three comparisons: the selected model FM v2 against SVD++, against FM v1, and against the content-based baseline. A result is reported as statistically significant if the t-test p-value falls below 0.05. The reporting of both parametric and non-parametric p-values follows the recommendation of the broader recommender-systems literature for rigorous comparative evaluation.

### 3.8.4 BENCHMARKING AGAINST TRADITIONAL MODELS

The final component of the evaluation strategy is the formal benchmarking of the proposed models against the traditional techniques that they are intended to supersede. The benchmark protocol is implicit in the design of the model family: the content-based baseline and the SVD++ baseline are trained and evaluated under *identical* conditions—the same data partitions, the same metric suite, and the same significance tests—as the culturally aware models.

This controlled equivalence is the cornerstone of the study's validity. Any observed difference between, for example, FM v2 and SVD++ can be attributed with confidence to the difference in their modelling assumptions—the presence or absence of cultural features—because every other variable is held constant. The benchmark therefore establishes not merely that the proposed system performs well in absolute terms, but that it performs *better than the culturally agnostic state of the practice*, and that the improvement is statistically significant and reproducible across multiple data splits.

The benchmarking is reported in two forms in Chapter 4: a comprehensive tabulation of all nine metrics for all models (with mean and standard deviation), and a set of comparative visualisations—grouped bar charts for error and ranking metrics, a radar chart overlaying the models across six performance axes, and a stratified trajectory plot showing the crossover between FM v2 and SVD++ as interaction depth grows. The hybrid engine, once its hyperparameters are tuned on the validation partition, is appended to the benchmark to demonstrate that it outperforms both of its constituent models, closing the evaluation loop between the individual models of Section 3.6 and the integrated system of Section 3.7.
