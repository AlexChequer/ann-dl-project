---
project: eda
task: regression
dataset: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
team:
  - Alex Chequer
  - Lucas Ikawa
  - Carlos Hernani Cardoso
ai_use: "Claude (Anthropic) helped write the scripts in code/ and draft the text; every number and decision was reviewed by the team."
---

# 1. EDA — New York City Airbnb 2019

!!! info "Team"

    | Full name | GitHub |
    |-----------|--------|
    | Alex Chequer | [AlexChequer](https://github.com/AlexChequer) |
    | Lucas Ikawa | [Lucasikawa](https://github.com/Lucasikawa) |
    | Carlos Hernani Cardoso | [carloshernanic](https://github.com/carloshernanic) |

    Dataset, decisions and status: [project page](../index.md).

**How to reproduce** (from the repository root, with `requirements.txt` installed):

```shell
python docs/projects/eda/code/eda_figures.py   # Stages 1–3: Figures 1–7 and eda_numbers.json
python docs/projects/eda/code/dimred.py        # Stage 4: Figures 8–10 and dimred_numbers.json
python docs/projects/eda/code/pipeline.py      # Stage 4C: shapes, NaN and feature names
```

Every number in the text comes from these scripts. `random_state=42` everywhere.

## 0. Proposal

| | |
|---|---|
| **Dataset** | New York City Airbnb Open Data (Inside Airbnb, via Kaggle — `AB_NYC_2019.csv`) |
| **Size** | 48,895 rows × 16 columns |
| **Task / target** | Regression — `price` (US$ per night) |
| **Features** | 7 numerical + 3 categorical used by the model (the other 5 columns are IDs, free text or a date) |
| **Motivation** | Predict a listing's nightly price from its location, room type and history. It mixes geography, high-cardinality categories and long-tailed counts. |
| **First risk** | Extremely skewed target (maximum US$ 10,000, median US$ 106). |

## 1. Initial inspection

### A. Data dictionary

**Source:** [Inside Airbnb](http://insideairbnb.com/), redistributed on Kaggle by
dgomonov (CC0 license). Snapshot of the active listings in New York City in 2019 (latest
review recorded on 2019-07-08).

**Each row** is one Airbnb listing in New York City.

| Feature | Meaning | Type | Unit |
|---------|---------|------|------|
| `id` | Unique identifier of the listing (one per row; 48,895 distinct) | ID (integer) | — |
| `name` | Title of the listing, written by the host (e.g. "Cozy room, great area, near train") | Free text | — |
| `host_id` | Unique identifier of the host; the same host can own several listings (37,457 distinct) | ID (integer) | — |
| `host_name` | First name of the host (11,452 distinct) | Free text (personal name) | — |
| `neighbourhood_group` | Borough of New York City where the listing is: Manhattan, Brooklyn, Queens, Bronx or Staten Island | Nominal categorical (5 levels) | — |
| `neighbourhood` | Neighbourhood inside the borough (e.g. Williamsburg, Harlem) | Nominal categorical (221 levels) | — |
| `latitude` | North–south coordinate of the listing (40.50 to 40.91) | Numeric continuous | decimal degrees |
| `longitude` | East–west coordinate of the listing (−74.24 to −73.71) | Numeric continuous | decimal degrees |
| `room_type` | Kind of space offered: Entire home/apt, Private room or Shared room | Nominal categorical (3 levels) | — |
| `price` | Nightly price set by the host — the **target** (0 to 10,000) | Numeric continuous (stored as integer) | US$ per night |
| `minimum_nights` | Minimum length of stay the host requires per booking (1 to 1,250) | Numeric discrete | nights |
| `number_of_reviews` | Total number of reviews the listing has received (0 to 629) | Numeric discrete (count) | reviews |
| `last_review` | Date of the most recent review (2011-03-28 to 2019-07-08); empty when the listing has no reviews | Date (`YYYY-MM-DD`) | — |
| `reviews_per_month` | Average rate of reviews received per month (0.01 to 58.5); empty when the listing has no reviews | Numeric continuous | reviews/month |
| `calculated_host_listings_count` | Number of listings the same host has in this dataset (1 to 327) | Numeric discrete (count) | listings |
| `availability_365` | Number of days the listing is open for booking in the next year (0 to 365) | Numeric discrete, bounded | days in the next 365 |

### B. Quality

**Missing values** (original CSV, 48,895 rows):

| Column | Missing | % | Pattern | Handling |
|--------|--------:|--:|---------|----------|
| `last_review` | 10,052 | 20.56% | missing only when `number_of_reviews == 0` | column dropped |
| `reviews_per_month` | 10,052 | 20.56% | same — **not random**: a listing with no reviews has no rate | impute **0** |
| `host_name` | 21 | 0.04% | random | column dropped |
| `name` | 16 | 0.03% | random | column dropped |

We checked that `reviews_per_month` is NaN **exactly** in the rows with
`number_of_reviews == 0` (they match in 100% of the rows). So a missing value means "zero
reviews per month", and 0 is the correct value, not an estimate.

**Duplicates:** 0 duplicated rows and 0 repeated `id`s.

**Impossible or inconsistent values:**

- `price == 0`: **11 rows**. A free night does not exist; since it is the target, these rows
  are removed (before the split, because the rule does not depend on any statistic).
- `minimum_nights > 365`: **14 rows** (maximum 1,250). A minimum stay longer than a year is
  inconsistent with short-term rental; the value is capped at 365.
- `availability_365 == 0`: **17,533 rows (35.9%)**. Not impossible (listing blocked or
  inactive), but it is a suspicious spike; kept as is and noted as a risk (Stage 5).

**Dropped columns:**

| Column | Reason |
|--------|--------|
| `id` | unique identifier, carries no information |
| `host_id` | identifier (37,457 distinct hosts); the useful information is already in `calculated_host_listings_count` |
| `name`, `host_name` | free text / personal name; out of scope for a tabular model |
| `last_review` | date with 20.56% missing; the review history is already in `number_of_reviews` and `reviews_per_month` |

**Leakage:** no column is derived from the price. `number_of_reviews`,
`reviews_per_month` and `availability_365` are observed **at the same time** as the price
(2019 snapshot), so they are available at prediction time. No feature has a suspicious
correlation with the target — the largest is |ρ| = 0.44 (`longitude`, Figure 4).

### C. Target

Statistics of `price` on the training set (39,107 listings):

| Mean | Median | Std | Min | Max | Skewness | Skewness of log1p |
|-----:|-------:|----:|----:|----:|---------:|------------------:|
| 153.30 | 106.00 | 245.16 | 10 | 10,000 | **19.22** | **0.64** |

![Target distribution](figures/fig01-target.png)
/// caption
**Figure 1** — Histogram of `price` and of `log1p(price)` on the training set.
///

The price has a very long right tail: the mean is 45% above the median and 2.17% of the
listings cost more than US$ 500. With `log1p`, the skewness drops from 19.22 to 0.64 and the
distribution becomes roughly symmetric. **Conclusion:** the network will predict
`log1p(price)`, otherwise the loss (MSE) is dominated by a few luxury listings.

### D. Train/test split

- 80% train / 20% test, `random_state=42`, on the 48,884 rows left after removing `price == 0`.
- **Stratified by `log1p(price)` quintile**: regression has no classes, so we cut the target
  into 5 bins so that train and test have the same price distribution (tail included).
- Not temporal: the dataset is a single 2019 snapshot, with no time axis for the target.
- **Train: 39,107 · Test: 9,777.** From here on, every statistic (Stages 2–4) is computed
  **on the training set only**.

## 2. Univariate analysis

### A. Numerical

| Feature | Mean | Median | Std | Q1 | Q3 | Min | Max | Skewness |
|---------|-----:|-------:|----:|---:|---:|----:|----:|---------:|
| `latitude` | 40.729 | 40.723 | 0.055 | 40.690 | 40.763 | 40.506 | 40.913 | 0.24 |
| `longitude` | −73.952 | −73.956 | 0.046 | −73.983 | −73.936 | −74.244 | −73.717 | 1.29 |
| `minimum_nights` | 6.97 | 3 | 17.65 | 1 | 5 | 1 | 365 | 11.95 |
| `number_of_reviews` | 23.25 | 5 | 44.26 | 1 | 24 | 0 | 629 | 3.69 |
| `reviews_per_month`* | 1.38 | 0.72 | 1.69 | 0.19 | 2.02 | 0.01 | 58.5 | 3.35 |
| `calculated_host_listings_count` | 7.02 | 1 | 32.41 | 1 | 2 | 1 | 327 | 8.03 |
| `availability_365` | 112.94 | 45 | 131.74 | 0 | 227 | 0 | 365 | 0.76 |

\* without the NaN (31,083 values).

![Numerical histograms](figures/fig02-numeric-hist.png)
/// caption
**Figure 2** — Histograms of the 7 numerical features on the training set. Features with
skewness > 3 are shown with a log-scale y axis.
///

- **Latitude/longitude:** almost symmetric and unimodal (concentrated in Manhattan/Brooklyn);
  the tail of `longitude` (skewness 1.29) is Queens, to the east. No transformation needed.
- **Long-tailed counts** (`minimum_nights`, `number_of_reviews`, `reviews_per_month`,
  `calculated_host_listings_count`): skewness between 3.35 and 11.95, median far below the
  mean and extreme outliers (e.g. one host with 327 listings). `minimum_nights` has modes at
  1–3 and a spike at **30** (NYC rule for long-term rentals).
- **`availability_365`:** bounded in [0, 365] and **bimodal** — a spike at 0 and another at 365.

**Conclusion:** the 4 counts need `log1p` + clipping so they do not dominate the scale; the
coordinates and `availability_365` only need standardization.

### B. Categorical

| Feature | Cardinality | Frequencies (train) |
|---------|------------:|---------------------|
| `room_type` | 3 | Entire home/apt 51.97% · Private room 45.69% · **Shared room 2.34%** |
| `neighbourhood_group` | 5 | Manhattan 44.28% · Brooklyn 41.10% · Queens 11.56% · **Bronx 2.30%** · **Staten Island 0.77%** |
| `neighbourhood` | **220** | top 5: Williamsburg 3,147, Bedford-Stuyvesant 2,981, Harlem 2,120, Bushwick 1,953, Upper West Side 1,617 |

![Categorical bars](figures/fig03-categorical-bars.png)
/// caption
**Figure 3** — Frequency of `neighbourhood_group`, `room_type` and of the 20 most common
neighbourhoods.
///

- **Rare categories:** `Shared room` (2.34%), `Bronx` (2.30%) and `Staten Island` (0.77%) have
  few examples — the model will see little of these groups.
- **High cardinality:** `neighbourhood` has 220 neighbourhoods in the training set, and **134
  of them have fewer than 50 listings** (together only 5.07% of the rows); 6 appear only
  once. A full one-hot would create 220 mostly empty columns.
- One test-set neighbourhood (**New Dorp**) does not appear in the training set.

**Conclusion:** `neighbourhood` needs rare categories grouped and new categories handled
(Stage 4A).

## 3. Bivariate and multivariate analysis

### A. Numerical × numerical

We use **Spearman** instead of Pearson: the counts have skewness up to 11.95 and extreme
outliers (Stage 2A), and Spearman measures monotonic relationships through ranks, without
being pulled by a few huge values. Example: between `number_of_reviews` and
`reviews_per_month`, Pearson gives 0.549 and Spearman 0.706.

![Spearman](figures/fig04-spearman.png)
/// caption
**Figure 4** — Spearman correlation matrix (7 numerical features + `price`), training set.
///

Most correlated pairs:

| Pair | Spearman ρ |
|------|-----------:|
| `number_of_reviews` × `reviews_per_month` | **0.706** |
| `calculated_host_listings_count` × `availability_365` | 0.407 |
| `reviews_per_month` × `availability_365` | 0.389 |

With the target, the largest is `longitude` (ρ = −0.44: further east, cheaper — leaving
Manhattan), followed by `latitude` (0.13). The review features barely relate to the price
(|ρ| ≤ 0.06).

**Conclusion:** the only **redundant** pair is `number_of_reviews` × `reviews_per_month`
(0.706). We keep both — 0.71 is not perfect collinearity and a network handles it well —
but it is noted. No pair goes above 0.8.

![Scatter](figures/fig05-scatter.png)
/// caption
**Figure 5** — Left: `number_of_reviews` × `reviews_per_month` (symlog scale). Right:
latitude × longitude colored by `log1p(price)`. Sample of 5,000 training listings.
///

**Conclusion:** on the left, the relationship is monotonic but wide (new listings have a high
rate with few total reviews), which confirms partial redundancy. On the right, the highest
prices concentrate in lower Manhattan — location is the most important numerical
information for the price.

### B. Categorical × target

![Target by category](figures/fig06-target-by-category.png)
/// caption
**Figure 6** — `log1p(price)` by `room_type` and by `neighbourhood_group`, training set
(outliers not drawn).
///

| `room_type` | Median (US$) | Mean | IQR |
|-------------|-------------:|-----:|----:|
| Entire home/apt | 160 | 212.2 | 110 |
| Private room | 70 | 90.6 | 46 |
| Shared room | 45 | 69.1 | 42 |

| `neighbourhood_group` | Median (US$) | Mean | IQR |
|-----------------------|-------------:|-----:|----:|
| Manhattan | 150 | 197.6 | 125 |
| Brooklyn | 91 | 125.6 | 90 |
| Queens | 75 | 98.8 | 60 |
| Staten Island | 75 | 102.7 | 65 |
| Bronx | 65 | 86.0 | 54 |

**Conclusion:** `room_type` is the categorical feature that separates price the most — the
median of an entire home (US$ 160) is more than twice that of a private room (US$ 70).
Among the boroughs, Manhattan is clearly above the other four, which overlap a lot.

### C. Numerical × categorical

![Numerical by room_type](figures/fig07-numeric-by-room-type.png)
/// caption
**Figure 7** — `availability_365`, `minimum_nights` and `calculated_host_listings_count` by
`room_type`, training set (outliers not drawn).
///

Each cell: median · IQR · std, training set. The IQR (Q3 − Q1) is computed with
`load_split()`; median and std come from `eda_numbers.json`.

| Feature | Entire home/apt (n = 20,322) | Private room (n = 17,868) | Shared room (n = 917) |
|---------|-----------------------------:|--------------------------:|----------------------:|
| `availability_365` (days) | 42 · 228 · 129.9 | 46 · 216 · 132.3 | **89 · 341** · 150.8 |
| `minimum_nights` (nights) | **3** · 3 · 19.8 | 2 · 3 · 14.7 | 1 · 2 · 16.7 |
| `calculated_host_listings_count` | 1 · 1 · **43.6** | 1 · 1 · 10.1 | **2 · 5** · 5.9 |

- **`availability_365`:** entire homes and private rooms are almost identical (median 42
  vs. 46 days, IQR 228 vs. 216, both with Q1 = 0), while shared rooms are open longer and
  vary more (median 89, IQR 341).
- **`minimum_nights`:** the median drops as the space gets less private (3 → 2 → 1 nights)
  and the IQR is narrow in all three groups (2–3 nights); the std (14.7–19.8) is 5–8× the
  IQR only because every group keeps the tail up to 365 nights.
- **`calculated_host_listings_count`:** entire homes and private rooms have median 1 and
  IQR 1 (mostly single-listing hosts), shared rooms median 2 and IQR 5. The std ranks the
  other way (43.6 > 10.1 > 5.9) because the largest hosts (up to 327 listings) offer entire
  homes and private rooms, while no shared-room host has more than 34.

**Conclusion:** `room_type` barely shifts these three features — entire homes and private
rooms overlap almost completely, differing by one night in median minimum stay — so they
carry information that `room_type` does not; only the rare shared room (2.34%) stands out,
with more availability, shorter minimum stays and larger hosts. For the two counts, the std
far above the IQR is the long tail again, which supports the clip + `log1p` of Stage 4A.

## 4. Preprocessing

### A. Strategies

Each decision comes from an earlier finding. Everything is fitted **on the training set
only** (`fit` on train, `transform` on test) and lives in [`code/pipeline.py`](#c-pipeline).

| # | Transformation | Features | Reason |
|---|----------------|----------|--------|
| 1 | Remove rows with `price == 0` (11) | target | impossible value (1B) |
| 2 | Cap `minimum_nights` at 365 (14 rows) | `minimum_nights` | inconsistent (1B) |
| 3 | Drop 5 columns | `id`, `host_id`, `name`, `host_name`, `last_review` | IDs/text/date (1B) |
| 4 | **Missing values:** impute **0** | `reviews_per_month` | NaN ⇔ zero reviews (1B) |
| 5 | **Outliers:** clip at the **train p99** | 4 counts | extreme tails (2A) |
| 6 | `log1p` | 4 counts | skewness 3.35–11.95 (2A) |
| 7 | **Scaling:** `StandardScaler` | all 7 numerical | very different scales; neural network |
| 8 | **Encoding:** one-hot, rare (< 50) → `infrequent`, unknown → `infrequent` | 3 categorical | 220 neighbourhoods, 134 rare, new neighbourhood in test (2B) |
| 9 | Target = `log1p(price)` | `price` | skewness 19.22 → 0.64 (1C) |

**Missing values.** Imputing 0 in `reviews_per_month` is not a guess: the value is missing
exactly when there are no reviews.

**Outliers.** We do not remove rows (they are real listings, and the network will meet them
in the test set). Instead, each count is clipped at the 99th percentile **of the training
set**: `minimum_nights` ≤ 45, `number_of_reviews` ≤ 214, `reviews_per_month` ≤ 6.86,
`calculated_host_listings_count` ≤ 232. This affects **1,345 training rows (3.44%)**. After
the clip, `log1p` compresses what is left of the tail.

**Encoding.** `room_type` and `neighbourhood_group` have few categories → plain one-hot.
For `neighbourhood`, we use `OneHotEncoder(min_frequency=50,
handle_unknown="infrequent_if_exist")`: the 134 neighbourhoods with fewer than 50 training
listings become a single `neighbourhood_infrequent_sklearn` column, and a neighbourhood that
only appears in the test set (like **New Dorp**) falls into that same column instead of
breaking the pipeline or becoming an all-zero vector.

**Scaling.** A neural network trains poorly with inputs on different scales (latitude varies
by 0.05; `availability_365` by 130). `StandardScaler` gives each numerical feature mean 0 and
standard deviation 1 on the training set. The one-hot columns stay in {0, 1}, already in the
right range.

### B. Dimensionality reduction

Projections of the **102 already-preprocessed features** of the training set, colored by
`log1p(price)`. PCA uses the whole training set; t-SNE and UMAP use a sample of **4,000
listings** (computational cost), the same sample in all three figures.

![PCA](figures/fig08-pca.png)
/// caption
**Figure 8** — Left: cumulative explained variance. Right: PC1 × PC2.
///

**PCA.** PC1 explains **22.37%** and PC2 **17.76%** — together, **40.14%**. **11 components**
are needed to reach 90%.

| Component | Largest loadings | Interpretation |
|-----------|------------------|----------------|
| PC1 | `reviews_per_month` (+0.61), `number_of_reviews` (+0.59), `minimum_nights` (−0.40) | **listing activity**: many reviews and short stays vs. few reviews and long stays |
| PC2 | `availability_365` (+0.62), `calculated_host_listings_count` (+0.60), `minimum_nights` (+0.38) | **"professional" profile**: host with many listings, always available |

The first two components only capture numerical listing-behavior features, which barely
relate to the price (Figure 4). That is why the colors in Figure 8 are mixed: **PCA does not
separate price** in 2D.

![t-SNE](figures/fig09-tsne.png)
/// caption
**Figure 9** — t-SNE with `perplexity` = 5 and 50.
///

![UMAP](figures/fig10-umap.png)
/// caption
**Figure 10** — UMAP with `n_neighbors` = 5 and 50 (`min_dist=0.1`).
///

**t-SNE and UMAP.** With `perplexity=5`, t-SNE only preserves very local neighborhoods and
becomes a cloud with no structure; with `perplexity=50`, well-defined groups appear. In UMAP,
`n_neighbors=5` produces many small fragments, and `n_neighbors=50` merges everything into a
few large groups. We checked the composition of the groups (k-means with k = 6 on the UMAP
with `n_neighbors=50`): each group is essentially **a single borough**
(`neighbourhood_group`), while `room_type` is mixed inside all of them. In other words, the
structure comes from the location one-hot columns. Within the groups, the color (price)
varies a lot.

**Comparison.** PCA is linear and favors the direction of **largest variance**, which here is
review behavior, not price. t-SNE and UMAP preserve **neighborhoods**, so they reveal that
the listings form discrete groups by borough — structure that PCA spreads out. But even
these groups do not separate price on their own: price also depends on `room_type`, which is
mixed inside each group (Figure 6). Caution: in t-SNE/UMAP, the **size** of the groups and
the **distance** between them have no direct meaning; only "who is close to whom" counts.

### C. Pipeline

`Pipeline` + `ColumnTransformer` in
[`docs/projects/eda/code/pipeline.py`](https://github.com/AlexChequer/ann-dl-project/blob/main/docs/projects/eda/code/pipeline.py),
importable from the repository root:

```python
from docs.projects.eda.code.pipeline import load_split, build_preprocessor, transform_target
X_train, X_test, y_train, y_test = load_split()
pre = build_preprocessor().fit(X_train)          # fit on train only
Xt_train, Xt_test = pre.transform(X_train), pre.transform(X_test)
y_train_model = transform_target(y_train)        # log1p(price)
```

??? example "code/pipeline.py"

    ```python
    --8<-- "docs/projects/eda/code/pipeline.py"
    ```

**Result** (`python docs/projects/eda/code/pipeline.py`):

- Remaining NaN: **0** in train and **0** in test.
- Final shape: train **(39,107, 102)**, test **(9,777, 102)**.
- 102 features = **7 numerical** + **95 one-hot**: 5 from `neighbourhood_group`, 87 from
  `neighbourhood` (86 neighbourhoods with ≥ 50 listings + `neighbourhood_infrequent_sklearn`)
  and 3 from `room_type`.

Names: `latitude`, `longitude`, `minimum_nights`, `number_of_reviews`,
`reviews_per_month`, `calculated_host_listings_count`, `availability_365`,
`neighbourhood_group_{Bronx, Brooklyn, Manhattan, Queens, Staten Island}`,
`neighbourhood_{Arverne, Astoria, …, Woodside}` (86),
`neighbourhood_infrequent_sklearn`,
`room_type_{Entire home/apt, Private room, Shared room}`. The full list is printed by the
script.

## 5. Synthesis

### Main findings

1. **The target needs `log1p`.** Skewness 19.22 → 0.64 (Figure 1).
2. **Location and room type drive the price.** `longitude` has ρ = −0.44 with price
   (Figure 4) and lower Manhattan concentrates the expensive listings (Figure 5); an entire
   home has a median of US$ 160 against US$ 70 for a private room (Figure 6).
3. **Review features barely explain price** (|ρ| ≤ 0.06, Figure 4), but they dominate the
   variance — that is why PC1 + PC2 (40.14%) do not separate price (Figure 8).
4. **Informative missingness:** `reviews_per_month` is missing exactly when there are no
   reviews (table in 1B) → impute 0.
5. **High-cardinality neighbourhoods:** 220 neighbourhoods, 134 with < 50 listings (Figure 3)
   → group the rare ones.
6. **Nonlinear structure:** t-SNE/UMAP show groups by borough that PCA does not show
   (Figures 8–10), but no 2D projection separates price — the model will have to combine
   location and `room_type` nonlinearly (MLP).

### Risks and plan

| Risk | Plan |
|------|------|
| Price tail (up to US$ 10,000) dominates the loss | train on `log1p(price)`; also report MAE in US$ |
| Little information to predict price (largest ρ is 0.44) | accept a high error; compare against a baseline that predicts the training median |
| 35.9% with `availability_365 == 0` (inactive listings?) | keep; in deliverable 2, test a binary `available == 0` flag |
| Rare categories (Shared room 2.34%, Staten Island 0.77%) | report the error per group in deliverable 2 |
| New neighbourhoods in test/production | already handled: `handle_unknown="infrequent_if_exist"` |
| `name` text dropped (may carry information: "luxury", "studio") | out of scope for now; possible extension |

### Results summary

| # | Item | Value |
|---|------|-------|
| 1 | Dataset, task and target | NYC Airbnb 2019 · regression · `price` (modeled as `log1p(price)`) |
| 2 | Instances × features (num / cat) | 48,895 × 16 in the CSV; the model uses 7 numerical / 3 categorical |
| 3 | Column with the most missing values | `last_review` and `reviews_per_month` — 20.56% (10,052) |
| 4 | Dropped columns and reason | `id`, `host_id` (IDs), `name`, `host_name` (text), `last_review` (date, 20.56% NaN, redundant) |
| 5 | Target mean and median | US$ 153.30 and US$ 106.00 (train) |
| 6 | Train / test size | 39,107 / 9,777 |
| 7 | Most correlated numerical pair | `number_of_reviews` × `reviews_per_month`, ρ = 0.706 |
| 8 | Rows affected by the outlier strategy | 1,345 training rows (3.44%) clipped at p99 |
| 9 | Variance of PC1 + PC2 | 40.14% |
| 10 | Shapes after the pipeline | train (39,107, 102) · test (9,777, 102) |
