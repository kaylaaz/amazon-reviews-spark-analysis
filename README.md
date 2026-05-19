# Milestone 3: Preprocessing & First Distributed Model with Amazon US Customer Reviews
---

## 1. Complete Preprocessing using Spark (8 points)

The preprocessing pipeline transforms the raw 109M-row Amazon US Customer Reviews dataset into a clean, model-ready format using Spark DataFrame operations and Spark MLlib transformers. The pipeline is structured in four modular stages for reproducibility.

**Stage 1 — Data Cleaning (`cleansing1`)**
Invalid product categories, malformed star ratings (non-1~5), null records, and duplicate review IDs are removed. Missing values in `star_rating`, `helpful_votes`, `total_votes` are filled using `Imputer` with median strategy.

**Stage 2 — Numerical Encoding (`cleansing2`)**
`verified_purchase` and `product_category` are converted to numeric indices using `StringIndexer`. Review headline character count is computed as a new feature (`review_headline_length`).

**Stage 3 — Text Analytics & Feature Engineering (`cleansing3`)**
Review body character count (`review_length`) is extracted. Log-transformed vote counts (`log_helpful_votes`, `log_total_votes`, `log_review_length`) are created. Review body is tokenized via `Tokenizer` for potential NLP use.

**Stage 4 — Target Engineering (`cleansing4`)**
Reviews with zero total votes are filtered. Binary label is defined as `label = 1` if `helpful_votes / total_votes ≥ 0.5`, else `0`. Stratified sampling by star rating (`{1:0.6, 2:1.0, 3:0.67, 4:0.32, 5:0.09}`) balances class distribution. Final 5 features are assembled via `VectorAssembler`.

**Result:** 109,830,520 → 11,405,130 rows (89.62% noise removed)
**Label distribution:** Label 0: 3,330,981 / Label 1: 8,074,149 (~2.4× imbalance)
**Features:** `star_rating`, `review_length`, `review_headline_length`, `verified_purchase_idx`, `category_idx`
**Split:** 70/15/15 (train/val/test, seed=42)

---

## 2. Train Your First Distributed Model (8 points)

Models were trained on **SDSC Expanse** using Spark MLlib with the following configuration:

```
spark.executor.instances = 15
spark.executor.memory    = 8g
spark.driver.memory      = 4g
spark.sql.shuffle.partitions = 400
```

### Model 1: Random Forest (numTrees=20)

```python
rf20 = RandomForestClassifier(featuresCol="features", labelCol="label",
                               numTrees=20, maxBins=64, seed=42)
```

| Split | AUC-ROC | F1 | Accuracy |
|-------|---------|----|----------|
| Train | 0.6727 | 0.6406 | 0.7208 |
| Val   | 0.6733 | 0.6408 | — |
| Test  | 0.6723 | 0.6407 | 0.7208 |

### Model 2: Random Forest (numTrees=50)

```python
rf50 = RandomForestClassifier(featuresCol="features", labelCol="label",
                               numTrees=50, maxBins=64, seed=42)
```

| Split | AUC-ROC | F1 | Accuracy |
|-------|---------|----|----------|
| Train | 0.6684 | 0.6431 | 0.7212 |
| Val   | 0.6692 | 0.6434 | — |
| Test  | 0.6683 | 0.6433 | 0.7213 |

### Model 3: GBT (maxIter=20, maxDepth=3)

```python
gbt = GBTClassifier(featuresCol="features", labelCol="label",
                    maxIter=20, maxDepth=3, maxBins=64, stepSize=0.1, seed=42)
```

| Split | AUC-ROC | F1 | Accuracy |
|-------|---------|----|----------|
| Train | 0.7008 | 0.6778 | 0.7295 |
| Val   | 0.7010 | 0.6778 | — |
| Test  | 0.7007 | 0.6778 | 0.7295 |

### Full Comparison Table

| Model | Train AUC | Test AUC | Train F1 | Test F1 | Train ACC | Test ACC |
|-------|-----------|----------|----------|---------|----------|----------|
| RF (numTrees=20) | 0.6727 | 0.6723 | 0.6406 | 0.6407 | 0.7208 | 0.7208 |
| RF (numTrees=50) | 0.6684 | 0.6683 | 0.6431 | 0.6433 | 0.7212 | 0.7213 |
| **GBT (iter=20, depth=3)** | **0.7008** | **0.7007** | **0.6778** | **0.6778** | **0.7295** | **0.7295** |

### Example Ground Truth and Predictions — RF (numTrees=20)

**Train Set:**
| label | prediction | star_rating | review_length | review_headline_length | verified_purchase_idx | category_idx |
|-------|------------|-------------|---------------|------------------------|----------------------|--------------|
| 0.0 | 0.0 | 1 | 1 | 1 | 0.0 | 24.0 |
| 1.0 | 0.0 | 1 | 1 | 8 | 0.0 | 0.0 |
| 0.0 | 0.0 | 1 | 2 | 2 | 0.0 | 3.0 |
| 0.0 | 0.0 | 1 | 2 | 2 | 0.0 | 5.0 |
| 0.0 | 0.0 | 1 | 2 | 8 | 0.0 | 3.0 |

**Validation Set:**
| label | prediction | star_rating | review_length | review_headline_length | verified_purchase_idx | category_idx |
|-------|------------|-------------|---------------|------------------------|----------------------|--------------|
| 0.0 | 0.0 | 1 | 1 | 8 | 0.0 | 2.0 |
| 1.0 | 0.0 | 1 | 2 | 8 | 0.0 | 5.0 |
| 0.0 | 0.0 | 1 | 3 | 4 | 0.0 | 2.0 |
| 0.0 | 0.0 | 1 | 3 | 8 | 0.0 | 0.0 |
| 0.0 | 0.0 | 1 | 3 | 8 | 0.0 | 0.0 |

**Test Set:**
| label | prediction | star_rating | review_length | review_headline_length | verified_purchase_idx | category_idx |
|-------|------------|-------------|---------------|------------------------|----------------------|--------------|
| 1.0 | 0.0 | 1 | 2 | 8 | 0.0 | 3.0 |
| 0.0 | 0.0 | 1 | 2 | 8 | 0.0 | 5.0 |
| 0.0 | 0.0 | 1 | 3 | 8 | 0.0 | 7.0 |
| 1.0 | 0.0 | 1 | 4 | 8 | 0.0 | 5.0 |
| 0.0 | 0.0 | 1 | 4 | 8 | 0.0 | 18.0 |

---

## 3. Fitting Analysis (4 points)

### Where does the model fit on the fitting graph?

All three models show Train AUC ≈ Val AUC ≈ Test AUC (gap < 0.002), placing them in the **good fit** zone with no significant overfitting.

| Model | Diagnosis |
|-------|-----------|
| RF (numTrees=20) | Good fit — Test AUC 0.6723. Moderate performance; structured features alone have limited signal. |
| RF (numTrees=50) | Slight underfitting compared to RF20 — more trees doesn't contribute to the performance, it seems it's because 20 trees were enough for fitting. |
| **GBT (iter=20, depth=3)** | **Good fit** — Test AUC 0.7007, highest among all models. Sequential error correction gives clear advantage. |

### Hyperparameter Comparison: RF20 vs RF50

Counterintuitively, RF50 performs worse than RF20 (Test AUC 0.6683 vs 0.6727). It is not significant difference. 20 trees are enough for fitting the dataset.

### Which model performs best and why?

**GBT (maxIter=20, maxDepth=3)** is the best model with Test AUC 0.7007 (+0.028 over RF20). Gradient boosting corrects residual errors sequentially at each round, capturing non-linear interactions such as the interplay between star rating and review length that a single-pass RF ensemble misses. Shallow trees (maxDepth=3) prevent overfitting while still modeling meaningful patterns.

### Next models planned for Milestone 4

| Model | Reason |
|-------|--------|
| **GBT with more iterations** | Increase `maxIter` to 50-100 with `stepSize=0.05` to test if deeper boosting improves AUC further |
| **XGBoost** (`SparkXGBClassifier`) | More efficient implementation of gradient boosting with column/row subsampling — expected higher AUC than GBT |

---

## 4. Conclusion Section (5 points)

### What is the conclusion of the 1st model?

Using 5 structured features from the Amazon review dataset — star rating, review body length, headline length, verified purchase status, and product category — GBT (maxIter=20, maxDepth=3) achieves the best performance with **Test AUC 0.7007, F1 0.6778, and Accuracy 0.7295**. The model generalizes well (Train ≈ Val ≈ Test, gap is lower than 0.001), considering there is no overfitting. GBT's sequential error correction outperforms Random Forest's parallel ensemble approach on this task eventhough the gap is not large, still demonstrating that helpfulness prediction benefits from capturing non-linear feature interactions.

### What can be done to improve it?

1. **More boosting rounds**: Increasing `maxIter` from 20 to 50-100 with a smaller `stepSize` (0.05) could further reduce residual errors
2. **Additional feature engineering**: Log-transformed vote counts (`log_total_votes`)
3. **Richer text features**: Replace character-count `review_length` with TF-IDF vectors or sentence embeddings from `review_body` — semantic content is a stronger helpfulness signal than raw length

### How did distributed computing help?

Training on 36M rows (filtered and sampled from 109M) across multiple Spark executors on SDSC Expanse parallelizes the data scanning and tree-building stages of GBT. Without distributed execution, even loading the full 50GB parquet dataset would exceed single-machine memory. Spark MLlib's Pipeline API ensures the preprocessing and modeling steps are reproducible and scalable across the cluster. (However, since the SpeedUp efficiency was low, the distributed computing was not as critical as expected)

---

## 5. Speedup Analysis (5 points)

### Speedup Table

**GBT Model Training:**

| Executors | Time (sec) | Speedup | Efficiency |
|-----------|------------|---------|------------|
| 1 | 165.68 | 1.00× | 100% |
| 15 | 150.19 | 1.10× | 7.4% |

**Preprocessing Pipeline:**

| Executors | Time (sec) | Speedup | Efficiency |
|-----------|------------|---------|------------|
| 1 | 112.36 | 1.00× | 100% |
| 15 | 109.65 | 1.02× | 6.8% |

### Amdahl's Law Analysis

**GBT Model Training:**
speedup = 1.10
n = 15

p = (1 - 1 / speedup) / (1 - 1 / n) # output: 0.097
speedup_max = 1 / ((1 - p) + (p / n)) # output: 1.10x

Only about 9.7% of the GBT model training process is effectively parallelized. The remaining 90.3% operates as serial workload. 
The bottleneck may be that GBT is an inherently sequential model. Because each tree must be built iteratively based on the previous one, the workload is fundamentally serial and cannot effectively benefit from distributed computing.

**GPreprocessing Pipeline:**
speedup = 1.02
n = 15

p = (1 - 1 / speedup) / (1 - 1 / n) # Output: 0.021
speedup_max = 1 / ((1 - p) + (p / n)) # output: 1.02x

The parallelizable fraction here is even lower, at just 2.1%. This indicates that almost the entire preprocessing pipeline (97.9%). The preprocessing pipeline was optimized with minimal use of shuffle-heavy methods, meaning data movement didn't occured much.
