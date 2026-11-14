---
project: regression
dataset: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
team:
  - Alex Chequer
  - Lucas Ikawa
  - Carlos Hernani Cardoso
ai_use: "Claude (Anthropic) helped write the scripts in code/ and draft the text; every number and decision was reviewed by the team."
---

# 2. Regression — New York City Airbnb 2019

!!! info "Team"

    | Full name | GitHub |
    |-----------|--------|
    | Alex Chequer | [AlexChequer](https://github.com/AlexChequer) |
    | Lucas Ikawa | [Lucasikawa](https://github.com/Lucasikawa) |
    | Carlos Hernani Cardoso | [carloshernanic](https://github.com/carloshernanic) |

    Dataset, decisions and status: [project page](../index.md). Previous deliverable:
    [EDA](../eda/index.md).

**How to reproduce** (from the repository root, with `requirements.txt` installed):

```shell
python docs/projects/regression/code/baselines.py   # Stage 1: baselines_numbers.json
python docs/projects/regression/code/train.py       # Stages 2–3: Figures 1–2 and train_numbers.json
python docs/projects/regression/code/search.py      # Stage 4: Figure 3, search_numbers.json, best_config.json
python docs/projects/regression/code/evaluate.py    # Stage 5, development run on validation (writes nothing)
python docs/projects/regression/code/evaluate.py --split test  # Stage 5: the test run — Figures 4–5,
                                                               # test_numbers.json, test_predictions.csv
python docs/projects/regression/code/error_analysis.py  # Stage 6: Figures 6–8 and error_numbers.json
```

Every number in the text comes from these scripts. Seed 42 everywhere (split, weight
initialization and batch order); Stages 4 and 5 repeat every training with seeds 42 to 46,
on the same split.

## 0 · Starting point

**Pipeline.** The EDA pipeline is **imported, not copied**:
[`code/data.py`](https://github.com/AlexChequer/ann-dl-project/blob/main/docs/projects/regression/code/data.py)
imports `load_split`, `build_preprocessor` and `transform_target` from
[`docs/projects/eda/code/pipeline.py`](../eda/index.md#c-pipeline). Same dataset, same task
(regression on `price`), same 80/20 split with `random_state=42`, so the test set has the
same **9,777** listings as in the EDA.

??? example "code/data.py"

    ```python
    --8<-- "docs/projects/regression/code/data.py"
    ```

**What changed since the EDA, and why:**

| Change | Reason |
|--------|--------|
| A **validation set** is taken from the EDA training set: 20% of it, stratified by `log1p(price)` quintile (same rule as the train/test split), seed 42 | every choice in Stages 1–4 is made on validation, so the test set is used only once, in Stage 5 |
| The preprocessor is **fitted on the new, smaller training set** (31,285 rows), not on the 39,107 rows of the EDA | fitting on rows that are now validation would leak their statistics (p99 clips, scaler means, category frequencies) |
| The model sees **91 features instead of 102** | consequence of the line above: with fewer training rows, only 75 neighbourhoods (instead of 86) reach the 50-listing minimum; the others fall into `neighbourhood_infrequent_sklearn`. 91 = 7 numerical + 5 boroughs + 76 neighbourhood columns + 3 room types |
| The test set is only loaded with `load_data(with_test=True)` | prevents an accidental use of the test set before Stage 5 |

| Set | Rows | Mean price (US$) | Median price (US$) |
|-----|-----:|-----------------:|-------------------:|
| Train | **31,285** | 154.06 | 106 |
| Validation | **7,822** | 150.25 | 106 |
| Test | **9,777** | not looked at until Stage 5 | — |

There are 0 NaN after the pipeline in train and validation.

**Target.** The network predicts `log1p(price)` (EDA, Stage 1C: skewness 19.22 → 0.64).
Every metric below is computed **in US$**, after inverting with `expm1`.

**Main metric: MAE in US$**, on the original scale. Reasons: (1) it reads directly as "the
prediction is off by US$ X per night on average"; (2) the EDA showed a price tail up to
US$ 10,000, and RMSE and R² in US$ are dominated by those few listings, so they would mostly
measure the luxury tail and not the typical listing. RMSE and R² are still reported, as
required.

**Success criterion, set before training any network:** the network's **MAE on the test set
must be at least 5% lower than the MAE of the best baseline** on the same test set. Why 5%
and not more: the EDA found little signal (largest |ρ| with the target is 0.44) and the
strongest effects — room type and borough — are additive in log scale, which a linear model
already captures. What is left for the network is the nonlinear part: the price surface over
latitude × longitude and its interaction with room type (EDA, Figure 5).

## 1 · Baselines

Both baselines use **the same pipeline and the same target** as the network: fitted on
`log1p(price)` with the 91 features, predictions converted back to US$, evaluated on the
**validation** set.

- **Trivial:** `DummyRegressor(strategy="mean")` — predicts the training mean of
  `log1p(price)` for every listing.
- **Linear:** `Ridge(alpha=1.0)` — linear regression with L2, because the 76 one-hot
  neighbourhood columns are sparse and correlated with the borough columns.

??? example "code/baselines.py"

    ```python
    --8<-- "docs/projects/regression/code/baselines.py"
    ```

| Baseline (validation) | MAE (US$) | RMSE (US$) | R² (US$) | RMSE in log |
|-----------------------|----------:|-----------:|---------:|------------:|
| Dummy (mean) | 81.78 | 218.16 | −0.029 | 0.687 |
| **Ridge** | **57.85** | 200.96 | 0.127 | 0.458 |

**Reading.**

- Ridge cuts the MAE of the trivial baseline by **29.3%** (81.78 → 57.85): room type and
  location alone already explain a large part of the price, as the EDA suggested (Figure 6).
- In log scale Ridge is much better than its R² in US$ suggests: the RMSE in log drops from
  0.687 to 0.458, which is **55.5% of the variance of `log1p(price)` explained**
  (1 − 0.458² / 0.687²). The R² in US$ is only 0.127 because the squared error in US$ is
  dominated by the luxury tail, which no feature here explains. This confirms the choice of
  MAE as the main metric.
- The Dummy has a slightly negative R² in US$ (−0.029) because it predicts the mean of the
  log, which after `expm1` is below the arithmetic mean that R² uses as reference.
- **Best baseline: Ridge, MAE US$ 57.85 on validation.** This is the number the network has
  to beat.

## 2 · The network

MLP in **PyTorch**, in
[`code/model.py`](https://github.com/AlexChequer/ann-dl-project/blob/main/docs/projects/regression/code/model.py).

``` mermaid
flowchart LR
    x["Input<br/>91 features"] --> h1["Linear 128<br/>ReLU"]
    h1 --> h2["Linear 64<br/>ReLU"]
    h2 --> out["Linear 1<br/>(no activation)"]
```

| | Choice | Why |
|---|--------|-----|
| Hidden layers | 2 | one hidden layer can already bend the linear solution; the second lets the network combine location with room type, the interaction the EDA pointed to. Other depths are tested in Stage 4 |
| Widths | 128 → 64 | first layer wider than the input (91) so that the sparse one-hot columns are not squeezed right away; then it narrows towards the single output |
| Activation | ReLU | cheap, does not saturate for positive inputs, standard for tabular MLPs |
| Initialization | He (Kaiming) normal, biases at 0 | designed for ReLU: keeps the variance of the activations stable from layer to layer |
| Output | 1 linear unit | regression: `log1p(price)` is an unbounded real number, so no activation |
| Loss | Huber (δ = 1) on `log1p(price)` | see below |

**Loss, justified by the EDA outliers.** Even after the log, the target keeps a right tail
(skewness 0.64; 2.17% of the EDA training listings cost more than US$ 500) that the features
do not explain. With MSE, those few listings would produce the largest gradients. Huber is
quadratic for errors below 1 log unit (a factor of about 2.7 in price) and linear above
that, so ordinary listings are treated as in MSE and the tail is treated as in MAE — which
also matches the main metric.

**Size.** The network has **20,097 parameters** (91·128 + 128 = 11,776; 128·64 + 64 = 8,256;
64 + 1 = 65) for **31,285 training rows** — 0.64 parameter per row. It has far more capacity
than Ridge (92 parameters) and could overfit, so early stopping (Stage 3) and regularization
(Stage 4) are needed, but it still has fewer parameters than training rows.

??? example "code/model.py"

    ```python
    --8<-- "docs/projects/regression/code/model.py"
    ```

## 3 · Training

The loop is `train_model` in
[`code/train.py`](https://github.com/AlexChequer/ann-dl-project/blob/main/docs/projects/regression/code/train.py);
Stages 4 and 5 reuse the same function.

| | Choice | Why |
|---|--------|-----|
| Optimizer | Adam (`AdamW` with `weight_decay=0`) | adapts the step per parameter, which matters here: the one-hot columns of rare neighbourhoods get a gradient in very few batches, while the 7 numerical features get one in every batch. Using `AdamW` lets Stage 4 turn on weight decay without changing the code |
| Learning rate | 1e-3 | compared with 1e-2 and 1e-4 below (Figure 2) |
| Batch size | 256 | 123 updates per epoch on 31,285 rows: enough updates to converge in few epochs, with batches large enough for a stable curve |
| Maximum epochs | 200 | only an upper limit; early stopping decides |
| Early stopping | patience 15 on the **validation loss**, best weights restored | the validation loss oscillates between epochs (Figure 1), so stopping at the first bad epoch would stop on noise; 15 epochs separates noise from a real upward trend |

??? example "code/train.py"

    ```python
    --8<-- "docs/projects/regression/code/train.py"
    ```

![Learning curves](figures/fig01-learning-curves.png)
/// caption
**Figure 1** — Huber loss on `log1p(price)` per epoch, training and validation, base network
(no regularization; log-scale y axis). The dashed line marks the epoch with the best
validation loss.
///

**Reading Figure 1.**

- **It converges, and fast.** The training loss falls from 0.786 in epoch 1 to 0.099 in
  epoch 4; the validation loss goes from 0.152 to 0.096 in the same 4 epochs. After that the
  gains are in the third decimal place.
- **Best epoch: 18** (validation loss **0.0899**, training loss 0.0869). Training stopped at
  epoch **33**, after 15 epochs with no improvement, and the weights of epoch 18 were
  restored.
- **Overfitting starts right after epoch 18, and it is mild.** From epoch 18 to 33 the
  training loss keeps falling (0.0869 → 0.0832) while the validation loss rises
  (0.0899 → 0.0915): the gap between the curves grows from 0.003 to 0.008.
- The small gap says the base network is **not limited by overfitting**. Training and
  validation losses are almost the same, so what limits the model is the information in the
  features (the EDA already expected a high error), not the capacity of the network.

![Learning-rate comparison](figures/fig02-lr-comparison.png)
/// caption
**Figure 2** — Validation loss per epoch for three learning rates (same network, same seed;
log-scale y axis).
///

| Learning rate | Best epoch | Stopped at | Best validation loss | Validation MAE (US$) |
|--------------:|-----------:|-----------:|---------------------:|---------------------:|
| 1e-2 | 23 | 38 | 0.0910 | 55.91 |
| **1e-3** | **18** | **33** | 0.0899 | **55.57** |
| 1e-4 | 129 | 144 | 0.0890 | 55.80 |

**Reading Figure 2.** With 1e-2 the validation loss is noisy — after epoch 5 it still jumps
between 0.091 and 0.110, while with 1e-3 it stays between 0.090 and 0.094 — and its best
value is the worst of the three (0.0910). With 1e-4 the curve is smooth and reaches the lowest loss
(0.0890), but it needs **129 epochs** — 7 times more than 1e-3 — and its MAE in US$ is not
better (55.80 against 55.57). **1e-3** gives the best MAE for a fraction of the cost, so it
is the base value; the learning rate is searched again in Stage 4 together with
regularization.

**Base network against the baselines (validation):**

| Model | MAE (US$) | RMSE (US$) | R² (US$) | RMSE in log |
|-------|----------:|-----------:|---------:|------------:|
| Dummy (mean) | 81.78 | 218.16 | −0.029 | 0.687 |
| Ridge | 57.85 | 200.96 | 0.127 | 0.458 |
| **MLP, base** | **55.57** | 198.30 | 0.149 | 0.440 |

The base network has a validation MAE **3.9% lower** than Ridge (57.85 → 55.57). On the
training set its MAE is 57.37 — higher than on validation because the training set holds
more of the extreme listings (training RMSE 235.24 against 198.30), not because of
underfitting on validation. The 3.9% is **below the 5% criterion** of Stage 0: the base
network alone would not meet it. Whether the final network meets it is decided on the test
set, in Stage 5.

## 4 · Regularization and hyperparameters

Everything in this stage uses **train and validation only**:
[`code/search.py`](https://github.com/AlexChequer/ann-dl-project/blob/main/docs/projects/regression/code/search.py)
calls `load_data()` without the test set and trains every configuration with `train_model`
from Stage 3 (same Huber loss, batch 256, early stopping with patience 15).

**Five seeds per configuration, not one.** Stage 3 used a single seed. Repeating the base
network with seeds 42 to 46 (the seed changes weight initialization and batch order; the
split stays fixed) gives a validation MAE of **55.84 ± 0.21** (mean ± sample standard
deviation). Most of the differences between configurations below are of a few tenths of a
dollar — the same order as this standard deviation — so a single seed would rank noise.
Every configuration is therefore trained 5 times and ranked by its **mean validation MAE**.
(With seed 42 alone the base network gives 55.55 here against 55.57 in Stage 3, with the
same best and stopping epochs, 18 and 33: the difference is floating-point arithmetic across
machines.)

??? example "code/search.py"

    ```python
    --8<-- "docs/projects/regression/code/search.py"
    ```

### 4A · Each regularization technique alone

Same base network (128 → 64, learning rate 1e-3); **one change at a time**, two strengths
for weight decay and for dropout:

| Technique | Setting | Validation MAE (US$) | Δ vs base | Mean best epoch |
|-----------|---------|---------------------:|----------:|----------------:|
| None (base) | — | 55.84 ± 0.21 | — | 25.2 |
| Weight decay (AdamW) | 0.01 | 55.81 ± 0.18 | −0.02 | 25.2 |
| Weight decay (AdamW) | 0.1 | 55.65 ± 0.12 | −0.19 | 28.0 |
| **Dropout** | **0.1** | **55.39 ± 0.23** | **−0.44** | 65.2 |
| Dropout | 0.3 | 55.63 ± 0.25 | −0.21 | 96.6 |
| BatchNorm | after each linear layer | 55.93 ± 0.35 | +0.10 | 24.4 |

Why these settings: in `AdamW` the decay is decoupled from the gradient, and each step
shrinks the weights by a factor of `lr × weight_decay` (1e-5 or 1e-4 per step here), so
0.01 and 0.1 go from weak to moderate. Dropout 0.1 and 0.3 bracket the usual range for a
network this narrow (64 units in the last hidden layer).

**Reading.**

- **Dropout is the only technique with a clear effect on its own**: 0.1 lowers the MAE by
  0.44, about twice the seed standard deviation. It also makes training longer (best epoch
  65.2 against 25.2): the noise it injects slows down the fit to the training set, and the
  validation loss keeps improving for more epochs.
- **More dropout is worse on this small network** (0.3: −0.21 only): removing 30% of 64
  units takes away capacity the network needs. Section 4C shows that the same 0.3 works
  well once the network is wider.
- **Weight decay barely matters**: 0.01 changes the MAE by 0.02 (a tenth of the seed
  standard deviation); 0.1 lowers it by 0.19, still within the noise.
- **BatchNorm makes it slightly worse** (+0.10) and is the least stable of the six (std
  0.35). The pipeline already standardizes every input and the network has only two hidden
  layers, so there is no conditioning problem for BatchNorm to fix; what it adds is noise
  from the batch statistics.
- This agrees with Stage 3: the base network was **not limited by overfitting** (small gap
  between the training and validation curves), so regularizing it changes little. Where
  regularization matters is when the capacity grows: the 256 → 128 network without
  regularization is the **worst** of all configurations below (56.13 ± 0.40; it peaks at
  epoch 19.0), and with dropout 0.2 the same network drops to 55.38 ± 0.13.

### 4B · The `availability_365 == 0` flag promised in the EDA

The EDA found that 35.9% of the listings have `availability_365 == 0` and proposed testing a
binary flag for them. Here the flag is appended as a 92nd column (`add_availability_flag`);
it is 1 for **36.0%** of the training rows.

| Model (validation) | Without the flag | With the flag | Δ MAE |
|--------------------|-----------------:|--------------:|------:|
| Base network (5 seeds) | 55.84 ± 0.21 | 55.93 ± 0.28 | +0.09 |
| Ridge | 57.85 | 57.65 | −0.19 |

**It does not help the network** (+0.09, inside the seed standard deviation), so it is
**not used**. A plausible reason: `availability_365` is already an input, standardized, and
0 is its minimum value, so a single ReLU unit can already isolate the listings at 0. Ridge
cannot build that step on its own and gains 0.19. The baselines stay exactly as defined in
Stage 1; with the flag, Ridge would be 0.19 better, which would only make the 5% criterion
harder to meet.

### 4C · Search

**What varies:** learning rate (3e-4, 1e-3, 3e-3), width and depth (from one layer of 64 up to
1024 → 512 → 256 and four hidden layers), and regularization (dropout 0 to 0.4, weight decay
0.1, BatchNorm). **What stays fixed:** Huber loss, `AdamW`, batch 256, early stopping with
patience 15, at most 200 epochs. **Selection rule, fixed before running:** the lowest mean
validation MAE over the 5 seeds.

The search ran in **two rounds**, both on validation:

- **Round 1** (9 configurations), designed after 4A: wider and deeper networks combined with
  dropout, the learning rate around the best of them, and BatchNorm and weight decay as
  add-ons.
- **Round 2** (4 configurations): the best of round 1 was the widest and most regularized
  network of the grid (512 → 256 → 128, dropout 0.3) — at its edge — so the search went on
  in that direction: more dropout, weight decay on top, one more layer, double the width.

Together with the 6 configurations of 4A, **19 configurations** are ranked, each trained
with 5 seeds (95 trainings). The flag of 4B is shown at the end, outside the ranking.

| # | Configuration (hidden units · lr · regularization) | Origin | Parameters | Validation MAE (US$) | RMSE (US$) | R² (US$) | Mean best epoch |
|--:|-----------------------------------------------------|--------|-----------:|---------------------:|-----------:|---------:|----------------:|
| 1 | **512-256-128 · lr 0.001 · dropout 0.4** | round 2 | 211,457 | **55.09 ± 0.15** | 198.89 | 0.144 | 64.6 |
| 2 | 512-256-128 · lr 0.001 · dropout 0.3 | round 1 | 211,457 | 55.09 ± 0.16 | 198.53 | 0.147 | 56.6 |
| 3 | 256-128-64 · lr 0.001 · dropout 0.2 | round 1 | 64,769 | 55.17 ± 0.13 | 199.11 | 0.143 | 47.8 |
| 4 | 1024-512-256 · lr 0.001 · dropout 0.3 | round 2 | 750,593 | 55.23 ± 0.32 | 198.94 | 0.144 | 46.0 |
| 5 | 512-256-128 · lr 0.001 · dropout 0.3 · wd 0.1 | round 2 | 211,457 | 55.26 ± 0.22 | 198.01 | 0.152 | 47.8 |
| 6 | 256-128-64 · lr 0.001 · dropout 0.1 · BatchNorm | round 1 | 65,665 | 55.28 ± 0.14 | 197.81 | 0.154 | 53.6 |
| 7 | 512-256-128-64 · lr 0.001 · dropout 0.3 | round 2 | 219,649 | 55.32 ± 0.40 | 199.60 | 0.138 | 44.2 |
| 8 | 256-128 · lr 0.001 · dropout 0.2 | round 1 | 56,577 | 55.38 ± 0.13 | 197.81 | 0.154 | 82.4 |
| 9 | 128-64 · lr 0.001 · dropout 0.1 | 4A | 20,097 | 55.39 ± 0.23 | 198.63 | 0.147 | 65.2 |
| 10 | 256-128 · lr 0.001 · dropout 0.2 · wd 0.1 | round 1 | 56,577 | 55.50 ± 0.25 | 198.72 | 0.146 | 51.0 |
| 11 | 256-128 · lr 0.003 · dropout 0.2 | round 1 | 56,577 | 55.53 ± 0.26 | 198.61 | 0.147 | 38.2 |
| 12 | 256-128 · lr 0.0003 · dropout 0.2 | round 1 | 56,577 | 55.54 ± 0.32 | 198.99 | 0.144 | 74.0 |
| 13 | 128-64 · lr 0.001 · dropout 0.3 | 4A | 20,097 | 55.63 ± 0.25 | 198.54 | 0.147 | 96.6 |
| 14 | 128-64 · lr 0.001 · wd 0.1 | 4A | 20,097 | 55.65 ± 0.12 | 198.30 | 0.149 | 28.0 |
| 15 | 128-64 · lr 0.001 · wd 0.01 | 4A | 20,097 | 55.81 ± 0.18 | 198.10 | 0.151 | 25.2 |
| 16 | 128-64 · lr 0.001 (base) | base | 20,097 | 55.84 ± 0.21 | 198.03 | 0.152 | 25.2 |
| 17 | 64 · lr 0.001 | round 1 | 5,953 | 55.90 ± 0.16 | 198.41 | 0.149 | 49.8 |
| 18 | 128-64 · lr 0.001 · BatchNorm | 4A | 20,481 | 55.93 ± 0.35 | 198.66 | 0.146 | 24.4 |
| 19 | 256-128 · lr 0.001 | round 1 | 56,577 | 56.13 ± 0.40 | 198.05 | 0.152 | 19.0 |
| — | 128-64 · lr 0.001 + `availability_365 == 0` flag | 4B | 20,225 | 55.93 ± 0.28 | 198.42 | 0.149 | 25.0 |

![Search results](figures/fig03-search.png)
/// caption
**Figure 3** — Validation MAE (US$) of every configuration, mean ± standard deviation over
5 seeds, sorted from best (top) to worst. Colors mark where each configuration comes from.
Dashed line: Ridge on validation (57.85); dotted line: 5% below Ridge (54.96), the margin
the Stage 0 criterion asks for on the test set.
///

**Reading Figure 3 and the table.**

- **What wins is capacity *with* regularization.** The top 3 all have 3 hidden layers and
  dropout of 0.2 or more. Without regularization, more width hurts: 256 → 128 alone is last
  (56.13), and the same network with dropout 0.2 is eighth (55.38).
- **The learning rate of Stage 3 holds.** Around 256 → 128 with dropout 0.2, both 3e-3
  (55.53) and 3e-4 (55.54) are worse than 1e-3 (55.38).
- **Round 2 found a plateau, which is why the search stops there.** Going from dropout 0.3
  to 0.4 changes nothing (55.09 and 55.09). Doubling the width (55.23) or adding a fourth
  layer (55.32) does not help and makes the result less stable (std 0.32 and 0.40).
- **The choice is a tie.** By the rule, dropout 0.4 is first (55.086 against 55.095, see
  `search_numbers.json`), but the gap is about 0.01 for a standard deviation of 0.15. The
  two are equivalent and the rule decides.
- **Gain over the base network: US$ 0.75** (55.84 → 55.09). The chosen network has 211,457
  parameters for 31,285 training rows — more parameters than rows; dropout 0.4 and early
  stopping are what keep it from memorizing the training set.
- **Already on validation, the margin is short of 5%:** the chosen network is **4.77% below
  Ridge** (57.85 → 55.09), to the left of the dashed line but not of the dotted one. Before
  opening the test set, the expectation was therefore a borderline result.

**Chosen configuration**, saved in `code/best_config.json` with only the keys that change
with respect to `BASE_CONFIG`:

```json
{"hidden": [512, 256, 128], "dropout": 0.4}
```

That is, **3 hidden layers of 512 → 256 → 128 units** (ReLU, He initialization), **dropout
0.4** after each hidden layer, no weight decay, no BatchNorm; `AdamW` with learning rate
1e-3, batch 256, Huber loss, early stopping with patience 15.

## 5 · Test evaluation

Script:
[`code/evaluate.py`](https://github.com/AlexChequer/ann-dl-project/blob/main/docs/projects/regression/code/evaluate.py).

**How the test set was used — once.**

1. Everything was fixed **before** the test set was loaded: the configuration
   (`best_config.json`, Stage 4), the seeds (42 to 46), the metrics, the figures and the
   decision rule for the criterion (**mean test MAE over the 5 seeds** against the best
   baseline's test MAE).
2. The script was developed on validation: without arguments, `evaluate.py` runs the whole
   evaluation on the validation set and writes nothing. That run reproduced Stage 4 exactly
   (55.09 ± 0.15), which confirms that retraining is deterministic. The figure and CSV code
   was also checked on validation predictions.
3. Then the script ran **once** with `--split test` — the only time Stages 0 to 5 loaded
   the test set (`load_data(with_test=True)`). **Nothing was changed after it** — no
   configuration, seed, threshold or figure choice was influenced by the test set. Running
   the command again reproduces the same numbers; it is not a new decision.

**Protocol.** For each seed, the chosen network is trained from scratch on the **31,285
training rows**, with early stopping on validation — the same protocol under which it was
selected. It is not retrained on train + validation, because early stopping needs the
validation set and the number of epochs would otherwise have to be guessed. Dummy and Ridge
are refitted on the same training rows with their Stage 1 settings. All models are evaluated
on the same **9,777 test listings**.

??? example "code/evaluate.py"

    ```python
    --8<-- "docs/projects/regression/code/evaluate.py"
    ```

| Model (test set, 9,777 rows) | MAE (US$) | RMSE (US$) | R² (US$) | RMSE in log |
|------------------------------|----------:|-----------:|---------:|------------:|
| Dummy (mean) | 82.12 | 222.21 | −0.029 | 0.685 |
| Ridge (best baseline) | 58.13 | 203.52 | 0.137 | 0.453 |
| **MLP 512-256-128, dropout 0.4** (5 seeds) | **55.28 ± 0.08** | 202.18 ± 0.62 | 0.148 ± 0.005 | 0.436 ± 0.001 |

| Seed | Test MAE (US$) | Below Ridge | Best epoch | Stopped at |
|-----:|---------------:|------------:|-----------:|-----------:|
| 42 | 55.23 | 4.99% | 69 | 84 |
| 43 | 55.38 | 4.74% | 73 | 88 |
| 44 | 55.36 | 4.77% | 74 | 89 |
| 45 | 55.26 | 4.94% | 59 | 74 |
| 46 | 55.19 | 5.06% | 48 | 63 |

**Was the Stage 0 criterion met? No.** The criterion asks for a test MAE at least 5% below
the best baseline: 0.95 × 58.13 = **US$ 55.22**. The network's mean test MAE is **55.28**,
**4.90%** below Ridge — US$ 0.06 short of the threshold. Only one seed of the five (46,
5.06%) passes on its own; picking it would mean choosing a seed by looking at the test set,
so the decision stays on the mean, as fixed in advance. The criterion is not changed.

**Is the gap to Ridge larger than the variation between seeds? Yes, by far.** The network
is **US$ 2.85** better than Ridge in mean MAE, and the standard deviation across seeds is
**US$ 0.08** — the gap is **34.6 times** the seed variation, and every one of the 5 seeds
beats Ridge (the worst seed has 55.38 against 58.13). So the improvement is real with
respect to training randomness; it is only smaller than the 5% we had asked for. The seed
standard deviation measures only training randomness, not which listings happened to fall
in the test set.

**Validation and test agree.** On validation the chosen network was 4.77% below Ridge; on
test it is 4.90%. Selecting 1 configuration out of 19 on validation did not produce an
optimistic validation number that collapsed on test.

**RMSE and R² tell a smaller story than MAE.** In RMSE the network gains little over Ridge
(202.18 against 203.52) and R² in US$ stays low (0.148 against 0.137). Both are dominated by
the luxury tail, as anticipated in Stage 0: the gain of the network is in the typical
listing, which is what MAE measures. In log scale the gain is clearer (RMSE 0.436 against
0.453).

**Output for Stage 6.** `code/test_predictions.csv` has one row per test listing:
`row_index` (row of `data/AB_NYC_2019.csv`, 0-based, to join with the raw features),
`price_true`, `price_pred` (network, seed 42), `pred_ridge` and `pred_dummy`, all in US$.
`code/test_numbers.json` holds every number of this section.

![Predicted versus actual](figures/fig04-pred-vs-actual.png)
/// caption
**Figure 4** — Predicted × actual price on the test set (both axes in log scale), for the
network (seed 42, left) and for Ridge (right). Each point is a listing; the dashed line is a
perfect prediction.
///

**Reading Figure 4.**

- Both clouds are **flatter than the diagonal**: cheap listings are overpredicted and
  expensive ones underpredicted. That is what a model does when the features explain only
  part of the price — it pulls predictions towards the middle.
- Listings priced between US$ 1,000 and US$ 10,000 are predicted at a few hundred dollars by
  **both** models. The EDA had already found that no feature identifies them, so neither
  model can.
- The network's cloud reaches higher predictions than Ridge's (its highest test prediction is
  US$ 984.94): it can combine location and room type nonlinearly, while Ridge only adds
  their effects in log scale.

![Residuals versus predicted](figures/fig05-residuals.png)
/// caption
**Figure 5** — Residual (actual − predicted price, US$) × predicted price, test set, network
with seed 42. The y axis is linear between −100 and +100 and logarithmic beyond; the x axis
is logarithmic. Red: median residual per quintile of the predicted price; orange: mean
residual per quintile.
///

**Reading Figure 5.**

- **The network is unbiased for the typical listing.** The median residual per predicted
  quintile stays between **−2.98 and +3.59 US$**, and the overall median residual is
  **−1.00 US$**, with **48.9%** of the listings underpredicted. That is the expected
  behaviour of a model trained on `log1p(price)`: back on the US$ scale, its prediction
  behaves like a conditional median, which is what minimizes MAE.
- **The mean tells a different story:** the mean residual is **+26.39 US$**, and per quintile
  it grows from **+4.43** (cheapest predictions, up to US$ 66.16) to **+60.82** (most
  expensive, above US$ 174.71). A few listings far above their prediction pull the mean up —
  the same luxury tail that keeps RMSE and R² low.
- **The error grows with the predicted price** (the cloud opens to the right): the model is
  less certain in absolute dollars where prices are higher.
- The diagonal stripes are listings with **round prices** (US$ 100, 150, 200...): for a
  fixed actual price, the residual falls linearly as the prediction rises.

## 6 · Error analysis

The analysis runs on the **validation** set with the base network of Stage 3 (MAE
US$ 55.57, against US$ 57.85 for Ridge). The permutation importance below shuffles ten
columns five times each: that is a repeated probe, not a single evaluation, so it belongs
on validation — the test set is read once, in Stage 5.

Every subgroup is also measured against **Ridge** on the same rows. "The network errs more
here" only means something next to what the linear baseline does with the same listings.
**Bias** is the mean of (prediction − true price): negative means the network asks for less
than the listing charges.

??? example "code/error_analysis.py"

    ```python
    --8<-- "docs/projects/regression/code/error_analysis.py"
    ```

### A. Error by subgroup

| `room_type` | Rows | MAE (US$) | Ridge (US$) | Gain | Mean price | MAE / price | Bias (US$) |
|-------------|-----:|----------:|------------:|-----:|-----------:|------------:|-----------:|
| Entire home/apt | 4,086 | 76.63 | 79.30 | 3.4% | 205.54 | 37.3% | −32.19 |
| Private room | 3,551 | 32.43 | 34.27 | 5.4% | 90.63 | 35.8% | −15.65 |
| **Shared room** | **185** | 34.51 | 36.55 | 5.6% | 73.55 | **46.9%** | −20.81 |

| `neighbourhood_group` | Rows | MAE (US$) | Ridge (US$) | Gain | Mean price | MAE / price | Bias (US$) |
|-----------------------|-----:|----------:|------------:|-----:|-----------:|------------:|-----------:|
| Manhattan | 3,479 | 75.74 | 79.12 | 4.3% | 195.04 | 38.8% | −33.33 |
| Brooklyn | 3,207 | 41.52 | 43.05 | 3.5% | 120.38 | 34.5% | −17.27 |
| Queens | 884 | 34.14 | 35.78 | 4.6% | 100.52 | 34.0% | −19.21 |
| **Bronx** | **196** | 28.24 | 27.35 | **−3.2%** | 81.15 | 34.8% | −7.48 |
| **Staten Island** | **56** | 40.72 | 38.92 | **−4.6%** | 105.20 | 38.7% | −20.47 |

| `availability_365` | Rows | MAE (US$) | Ridge (US$) | Gain | Mean price | MAE / price | Bias (US$) |
|--------------------|-----:|----------:|------------:|-----:|-----------:|------------:|-----------:|
| `== 0` | 2,787 | 43.20 | 45.36 | 4.8% | 128.96 | 33.5% | −16.42 |
| `> 0` | 5,035 | 62.41 | 64.76 | 3.6% | 162.04 | 38.5% | −28.83 |

![MAE by subgroup](figures/fig06-mae-by-subgroup.png)
/// caption
**Figure 6** — MAE in US$ by room type and by borough, validation set. Bars are the
network, red diamonds are Ridge on the same rows, and the dashed line is the MAE over all
7,822 rows (US$ 55.57).
///

**Reading Figure 6.**

- **MAE in US$ tracks the price level, not rarity.** Entire home/apt has the largest MAE
  (76.63) only because it is the most expensive group (mean US$ 205.54); its *relative*
  error, 37.3%, is close to Private room's 35.8%. Reading the subgroups in US$ alone would
  put the blame on the biggest, best-represented group — which is why the table carries
  MAE / mean price.
- **Where rarity does show up is the relative error of Shared room: 46.9%**, 11 points
  above Private room (35.8%) at a comparable price level (US$ 73.55 against US$ 90.63).
  *Hypothesis:* with 2.34% of the training rows (EDA, Stage 2B) the network has little
  evidence for this level and falls back on something close to the room-type average. The
  price band of comparable listings (US$ 60–90) errs by 27.8%, so the 46.9% is not simply
  "cheap listings are harder".
- **The two rarest boroughs are the only groups where the network loses to Ridge:** Bronx
  −3.2% and Staten Island −4.6%, against gains of 3.4% to 5.6% everywhere else.
  *Hypothesis:* the network's gain comes from bending the price surface locally, which
  needs enough listings per region; in the two boroughs the EDA flagged as rare (2.30% and
  0.77% of the dataset) there is not enough local evidence, and the extra flexibility costs
  more than it earns. With 56 rows, Staten Island alone is US$ 1.80 of difference and is
  not reliable; the Bronx, with 196 rows, points the same way.
- **`availability_365 == 0` is not a harder subgroup.** The EDA flagged these 35.9% of
  listings as suspicious; they are 35.63% of validation, and they err *less*, both in US$
  (43.20 against 62.41) and relatively (33.5% against 38.5%). *Hypothesis:* the flag marks
  cheaper listings (mean US$ 128.96 against US$ 162.04) rather than harder ones. The column
  still carries signal — it is fourth in the permutation ranking below.

### B. Error by band of the true price

| Band of the true price | Rows | MAE (US$) | Ridge (US$) | Gain | Mean price | MAE / price | Bias (US$) |
|------------------------|-----:|----------:|------------:|-----:|-----------:|------------:|-----------:|
| US$ 10–60 | 1,608 | 16.11 | 19.87 | **18.9%** | 47.15 | 34.2% | **+14.30** |
| US$ 60–90 | 1,574 | 21.29 | 23.07 | 7.7% | 76.56 | 27.8% | +7.40 |
| US$ 90–130 | 1,569 | 32.34 | 35.88 | 9.9% | 110.06 | 29.4% | +11.99 |
| US$ 130–200 | 1,730 | 41.38 | 41.77 | 0.9% | 167.81 | 24.7% | −6.35 |
| US$ 200–9,999 | 1,341 | 188.59 | 190.65 | 1.1% | 384.76 | 49.0% | −174.06 |
| *tail: > US$ 500* | *166* | **768.74** | 793.06 | 3.1% | 1,003.86 | **76.6%** | **−768.74** |

The last row is the luxury tail, already contained in the top band; it is listed apart
because the EDA singled it out.

![MAE by price band](figures/fig07-mae-by-price-band.png)
/// caption
**Figure 7** — MAE in US$ by band of the true price, validation set (bars: network;
diamonds: Ridge; green line, right axis: MAE as a percentage of the band's mean price).
Bands are the quintiles of the true price; the last column is the tail above US$ 500.
///

**Reading Figure 7.**

- **The network under-predicts, and the bias grows with the price.** The bias goes from
  **+14.30** in the cheapest band to **−174.06** in the top one. In the tail it is
  **−768.74 — exactly minus the MAE**, which can only happen if *every one of the 166
  listings above US$ 500 is predicted below its price*.
- *Hypothesis:* the target is `log1p(price)` under a Huber loss, so a listing priced at
  US$ 1,000 predicted at US$ 300 costs about the same as one at US$ 100 predicted at
  US$ 30. With features that do not explain the tail (EDA: largest |ρ| is 0.44), the
  cheapest thing the optimizer can do is sit near the conditional median of the log, which
  in US$ is far below the tail. This is the same argument that made MAE the main metric in
  Stage 0, now visible per band.
- **The network's gain over Ridge lives in the cheap bands and vanishes at the top:**
  18.9% in US$ 10–60 and 9.9% in US$ 90–130, against 0.9% and 1.1% in the two most
  expensive bands. The nonlinearity the network adds is worth real percentage points where
  the prices are small, and almost nothing where the dollars are.
- The U shape of the relative error (34.2% → 24.7% → 49.0% → 76.6%) says the model is at
  its best in the US$ 130–200 band, which is where the bias crosses zero.

### C. The ten worst predictions

| Row | True (US$) | Predicted (US$) | Error | Room type | Borough · neighbourhood | Min. nights | Reviews | Availability |
|----:|-----------:|----------------:|------:|-----------|-------------------------|------------:|--------:|-------------:|
| 12342 | 9,999 | 82.8 | 9,916.2 | Private room | Manhattan · Lower East Side | 99 | 6 | 83 |
| 44034 | 6,800 | 537.6 | 6,262.4 | Entire home/apt | Manhattan · Chelsea | 1 | **0** | 364 |
| 2698 | 5,000 | 371.1 | 4,628.9 | Entire home/apt | Manhattan · Nolita | 1 | 2 | 365 |
| 29663 | 3,518 | 326.2 | 3,191.8 | Entire home/apt | Manhattan · East Village | 1 | **0** | 215 |
| 28727 | 3,000 | 75.6 | 2,924.4 | Private room | Brooklyn · Brighton Beach | 1 | **0** | 90 |
| 38498 | 2,999 | 189.9 | 2,809.1 | Entire home/apt | Manhattan · Upper West Side | 1 | **0** | 0 |
| 46533 | 2,995 | 206.6 | 2,788.4 | Entire home/apt | Manhattan · Chelsea | 30 | 1 | 214 |
| 22992 | 2,900 | 313.8 | 2,586.2 | Entire home/apt | Manhattan · Upper West Side | 1 | **0** | 180 |
| 29665 | 2,626 | 227.3 | 2,398.7 | Entire home/apt | Brooklyn · Park Slope | 1 | **0** | 31 |
| 45710 | 2,500 | 192.2 | 2,307.8 | Entire home/apt | Manhattan · Hell's Kitchen | 70 | **0** | 191 |

Row numbers are lines of `data/AB_NYC_2019.csv`, counting from 0.

**Reading.** All ten are under-predictions, all cost at least US$ 2,500, and the network
never goes above US$ 537.6 on any of them. Two patterns, each with a number behind it:

- **Seven of the ten have zero reviews**, against 20.56% of the dataset (EDA, Stage 1B:
  `last_review` is missing exactly when `number_of_reviews == 0`). *Hypothesis:* these are
  not listings the market priced — they are placeholders or blocked calendars, where the
  host types a deterrent price. The dataset has no "was it ever booked" column, so nothing
  in the features can separate them from a real luxury listing, and no model trained on
  these ten features will predict them.
- **The extreme case is row 12342: a Private room in Lower East Side at US$ 9,999 with a
  99-night minimum.** A private room at ten thousand dollars a night with a three-month
  minimum stay is not a product; the network predicts US$ 82.8, which is the right answer
  for the features it can see.
- Eight of the ten are in Manhattan and eight are Entire home/apt — the two levels with the
  highest mean price, which is where a tail can exist at all.

### D. Permutation importance

![Permutation importance](figures/fig08-permutation-importance.png)
/// caption
**Figure 8** — Increase in validation MAE when one original column is shuffled, mean of 5
shuffles (error bars: ± 1 standard deviation). The 10 original columns are shuffled, not
the 91 model columns, so one shuffle moves every one-hot column the feature produces.
///

| Original column | Δ MAE (US$) | ± std | EDA, Spearman with price |
|-----------------|------------:|------:|-------------------------:|
| `room_type` | **25.43** | 0.38 | categorical |
| `longitude` | 9.57 | 0.31 | −0.439 |
| `minimum_nights` | 7.56 | 0.17 | 0.102 |
| `availability_365` | 7.02 | 0.27 | 0.086 |
| `neighbourhood` | 5.98 | 0.32 | categorical |
| `latitude` | 4.83 | 0.18 | 0.132 |
| `calculated_host_listings_count` | 3.59 | 0.25 | −0.108 |
| `number_of_reviews` | 2.90 | 0.17 | −0.057 |
| `neighbourhood_group` | 2.40 | 0.12 | categorical |
| `reviews_per_month` | 1.32 | 0.10 | −0.021 |

**Does it confirm the EDA? Yes, with one caveat about how the question is asked.**

- **`room_type` dominates: +25.43, 2.7 times the next column.** The EDA concluded that
  `room_type` is "the categorical feature that separates price the most" (Stage 3B). The
  network agrees, and by a wide margin.
- **Location is confirmed as well, but it is split over four columns.** The EDA called
  location "the most important numerical information for the price" (Stage 3A, Figure 5).
  Here `longitude` leads the numerical columns at +9.57, as its |ρ| = 0.44 predicted — but
  `longitude`, `neighbourhood`, `latitude` and `neighbourhood_group` together sum to
  **+22.78**, close to `room_type` on its own. *Caveat:* permuting one column at a time
  splits the credit among correlated columns, so it understates any feature that the
  pipeline spreads out. `neighbourhood_group` is the clearest case: shuffling it costs only
  +2.40 because the 76 `neighbourhood` columns still carry the borough.
- **Reviews are last, as the EDA predicted, but not at zero.** `number_of_reviews` (+2.90)
  and `reviews_per_month` (+1.32) are the smallest numerical contributions, matching
  |ρ| ≤ 0.06 in the EDA. That they are not zero is consistent with Section C: zero reviews
  is a weak marker of a listing that was never transacted, and the network does use it.
- `minimum_nights` at +7.56 is the one ranking the EDA did not anticipate: its |ρ| is only
  0.102, the third smallest. A monotonic correlation does not see a feature that matters
  only through a nonlinear rule — a one-night minimum and a 99-night minimum describe
  different products at the same price point.

## 7 · Conclusion

**Was the network worth it?** On validation, against the best baseline: MAE US$ 55.57
against Ridge's US$ 57.85, **3.9% lower**. The cost of that 3.9% is 20,097 parameters
against 92, a PyTorch dependency, and a training loop with early stopping instead of a
closed-form fit. Section 6B says where the 3.9% comes from: **18.9% in the US$ 10–60 band
and 9.9% in US$ 90–130, against 0.9% and 1.1% in the two most expensive bands.** The
network buys accuracy where the prices are small in absolute terms, which is worth little
in dollars; on the listings where an error costs the most, it matches the linear model. And
on the two rarest boroughs it is *worse* than Ridge. The honest summary for this network is
that it is a marginal improvement that has not met the 5% criterion set in Stage 0.

**Limitations.**

- **The data is from 2019.** It is a pre-pandemic snapshot of a market that has since
  changed in level and in regulation; nothing here transfers to today's prices without
  being refitted.
- **The listing text was dropped.** `name` is the one column that carries words like
  "luxury", "penthouse" or "loft", which is exactly the information missing to explain the
  tail that Section 6B and 6C showed the network cannot reach.
- **There is no outcome column.** Nothing says whether a listing was ever booked, so a
  placeholder at US$ 9,999 and a real luxury listing are identical to the model. Seven of
  the ten worst errors are of this kind.
- **Prices are point predictions with no interval.** A model with 34% to 77% relative error
  depending on the band reports a single number, which hides how uncertain it is.

**Who is affected by an error.** The bias is not symmetric, so the harm is not either:

- **A host with an expensive listing is told to undercharge.** In the US$ 200–9,999 band
  the bias is −174.06 on a mean price of 384.76; a host who prices from this model gives up
  real revenue on every night booked.
- **A host with a cheap listing is told to overcharge**, by +14.30 on a mean price of 47.15
  in the bottom band — a 30% markup that can leave the calendar empty.
- **The smallest hosts are served worst.** The model is least reliable exactly where there
  are fewest listings: 46.9% relative error on Shared room, and worse than a linear model
  in the Bronx and on Staten Island. Any tool built on this would work better for the
  Manhattan segment than for the hosts with the least room to absorb a bad night.
- **A guest** reading a prediction as a fair price would judge most expensive listings
  overpriced, because the model is systematically below them.

**What we would do with more time.**

1. **Text features from `name`** (TF-IDF or sentence embeddings), the cheapest likely gain
   given that the tail is the main source of error.
2. **An "is it active" proxy** from reviews and availability, to flag or drop the
   placeholder listings that dominate the worst cases instead of training against them.
3. **Quantile regression** (or a two-stage model: classify "tail", then regress), to report
   an interval and to stop optimizing a single point for a distribution this skewed.
4. **Target encoding for `neighbourhood`** instead of 76 one-hot columns, which would also
   make the permutation importance of location readable in one number.

## Results summary

| # | Summary | Value |
|---|---------|-------|
| 1 | Dataset, task, target and main metric | NYC Airbnb 2019 · regression · `price` (modeled as `log1p(price)`) · MAE in US$ |
| 2 | Train · validation · test (rows) | 31,285 · 7,822 · 9,777 |
| 3 | Best baseline and its test metric | Ridge (`alpha=1.0`) · test MAE **US$ 58.13** (RMSE 203.52, R² 0.137) |
| 4 | Chosen network: layers × units, number of parameters, regularization | 3 hidden layers, 512 → 256 → 128 units (ReLU) · 211,457 parameters · dropout 0.4 + early stopping (no weight decay, no BatchNorm) |
| 5 | Optimizer, learning rate, batch and the epoch it stopped at | `AdamW` without weight decay (= Adam) · lr 1e-3 · batch 256 · best epoch 48 to 74 (mean 64.6), stopped at 63 to 89 (mean 79.6) over the 5 seeds |
| 6 | The network's main metric on the test set (mean ± std, 3 seeds) | MAE **US$ 55.28 ± 0.08** over 5 seeds (RMSE 202.18 ± 0.62, R² 0.148 ± 0.005) |
| 7 | Success criterion met? | **No** — 4.90% below Ridge, the criterion asked for 5% (threshold US$ 55.22); the US$ 2.85 gap to Ridge is 34.6 times the seed standard deviation |
| 8 | Subgroup with the largest error and its value | <!-- LUCAS, refeito após a etapa 5 --> |
