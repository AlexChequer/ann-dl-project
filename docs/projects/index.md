# Project

!!! abstract "Assignments"

    [Projects](https://insper.github.io/ann-dl/){:target='_blank'}

The project is **a single one**, done as a team on **the same dataset**, and delivered in
three parts over the semester.

## Team

| Full name | GitHub |
|-----------|--------|
| Alex Chequer | [AlexChequer](https://github.com/AlexChequer) |
| | [Lucasikawa](https://github.com/Lucasikawa) |
| Carlos Hernani Cardoso | [carloshernanic](https://github.com/carloshernanic) |

## The three deliverables

| # | Deliverable | Page |
|---|-------------|------|
| 1 | EDA | [EDA](eda/index.md) |
| 2 | Regression | [Regression](regression/index.md) |
| 3 | Generative | [Generative](generative/index.md) |

## Dataset

| | |
|---|---|
| **Name** | New York City Airbnb Open Data (`AB_NYC_2019.csv`) |
| **Source (URL)** | <https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data> (Inside Airbnb data) |
| **License / terms of use** | CC0 1.0 (public domain) |
| **Instances** | 48,895 listings (48,884 after removing `price == 0`) |
| **Features** | 16 columns in the CSV; 7 numerical + 3 categorical used by the model |
| **Target** | `price` — nightly price in US$ (modeled as `log1p(price)`) |
| **Chosen task** | Regression |

Predict a listing's nightly price from where it is, what kind of space it offers and its
review/availability history. It is not trivial: the target has a very long tail (up to
US$ 10,000), location has 220 neighbourhoods with many rare ones, and no single numerical
feature explains price well (largest |ρ| = 0.44). The CSV is versioned in `data/`.

## Status

- [x] **1. EDA**
- [ ] **2. Regression**
- [ ] **3. Generative**

## Decision log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-10-14 | Dataset NYC Airbnb 2019, regression on `price` | see EDA |
| 2026-10-14 | Target modeled as `log1p(price)` | skewness 19.22 → 0.64 |
| 2026-10-14 | Pipeline in `docs/projects/eda/code/pipeline.py` reused in the next deliverables | fit on train only, 102 features |
