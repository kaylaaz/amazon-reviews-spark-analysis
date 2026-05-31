# Amazon US Customer Reviews Spark Analysis

This project uses the [Amazon US Customer Reviews Dataset](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset) from Kaggle to investigate what makes a product review useful to other shoppers. The dataset is approximately 50.68 GB and contains 109,830,520 reviews written between 1995 and 2015 across dozens of product categories. Using Spark on the SDSC Expanse supercomputer for distributed processing, we explore the data, engineer features, and build distributed models that predict whether a review will be found helpful, which is defined as having a helpfulness ratio (`helpful_votes / total_votes`) of at least 0.5. Our first model is a distributed Random Forest trained on structured metadata, reaching a test AUC of ~0.71. Our second model applies PCA dimensionality reduction followed by a supervised classifier to test whether a compressed feature representation can match the full-feature model. It retains 70.9% of variance in three components but underperforms the full-feature models, demonstrating that dimensionality reduction discards signal when the input features are already low-dimensional and largely independent. The broader goal is to identify the signals that separate valuable reviews from those that are ignored.

---

## 1. Introduction

### Why This Project Was Chosen

Online reviews are one of the most influential factors in modern purchasing decisions, yet the vast majority of reviews receive little or no engagement. Amazon surfaces "helpful" reviews near the top of product pages, which means a small fraction of reviews shape what millions of shoppers see and decide to buy. Understanding what makes a review helpful is therefore both commercially and behaviorally interesting. Customer reviews are at the intersection of consumer psychology, text analysis, and large-scale data engineering.
 
We chose this dataset because it is large (over 109 million rows, ~50.68 GB) and combines structured metadata (star rating, votes, verified-purchase status, category) with unstructured text (review headline and body). This allows us to ask a focused and measurable question: which review attributes predict perceived helpfulness and which reviews are helpful based on the helpfulness ratio? 

### Broader Impact of a Good Predictive Model

A reliable model of review helpfulness has practical value for multiple parties. For shoppers, seeing genuinely helpful reviews reduces the time spent sifting through low-quality or spam content. For platforms, it improves trust and conversion by promoting reviews that other buyers actually rely on. For sellers, it clarifies what kind of feedback is most visible and useful. The same approach transfers to any domain where user-generated content must be ranked by usefulness (e.g., Q&A sites, support forums, recommendation systems).

### Why This Problem Requires Big Data and Distributed Computing

The full dataset is approximately 50.68 GB on disk and expands substantially in memory during Spark operations such as joins, aggregations, and model training. A single-machine workflow using pandas or scikit-learn would not be able to load, let alone process, 109 million rows without running out of memory. Several steps in our pipeline (e.g., counting and deduplicating 109M review IDs, computing missing values and distribution statistics across every column, aggregating helpfulness ratios by category and year, and training ensemble models on roughly 11 million cleaned rows) are only practical when the work is partitioned and executed in parallel across many cores.
 
Spark makes this feasible by distributing both data and computation. The dataset is split into partitions processed simultaneously across executors, and MLlib's training algorithms aggregate split statistics in a distributed fashion rather than scanning all data on one core. Without Spark (or a comparable framework such as Ray), iterating across multiple model configurations at this scale would take many hours to days on a single machine and would frequently fail on memory limits, making the hyperparameter comparison and dimensionality-reduction experiments in this project effectively impractical. Our speedup analysis quantifies the concrete benefit measured on Expanse.

---

## 2. SDSC Expanse Environment Setup

This project was developed and executed on the San Diego Supercomputer Center (SDSC) Expanse cluster. Given the dataset size, significant RAM and parallelism are required.

| Parameter | Value |
| --- | --- |
| Partition | shared |
| Type | JupyterLab |
| Total Cores | 16 |
| Total Memory | 128 GB |
| Driver Cores | 1 |
| Executor Instances | 15 |
| Executor Memory | 8 GB |

Formulas used:

- `Executor instances = Total Cores - 1` = 16 - 1 = 15
- `Executor memory = (Total Memory - Driver Memory) / Executor Instances` = (128 - 2) / 15 ≈ 8.4 GB (configured at 8 GB per executor)

Justification: 128 GB total memory across 16 cores balances volume and parallelism for this workload. The 50.68 GB dataset expands in memory during Spark operations, so allocating ~8 GB per executor prevents out-of-memory errors during aggregation-heavy operations and model training. Reserving one core and memory for the driver keeps the cluster stable while maximizing parallel task execution across 15 worker executors.

### Spark Session Configuration 

```python
spark = SparkSession.builder \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config('spark.executor.instances', 15) \
    .config("spark.executor.cores", "1") \
    .config("spark.sql.shuffle.partitions", "64") \
    .config("spark.local.dir", f"/expanse/lustre/projects/uci157/{USER}/spark_tmp") \
    .config("spark.sql.parquet.columnarReaderBatchSize", "128") \
    .getOrCreate()
```

### Spark UI Screenshot
<img width="523" height="57" alt="Screenshot" src="https://github.com/user-attachments/assets/b72c09b8-759e-4806-9b4c-eaa612b78c60" />

Figure 1. Spark UI captured during data loading, which shows multiple active executors on SDSC Expanse and confirming the job ran in a distributed configuration rather than locally.

---

## 3. Methods

This section summarizes our methods in the order they were executed. The reasoning behind each choice is discussed in the Discussion section.

### 3.1 Data Exploration

All exploration was performed using Spark DataFrames on the full 109,830,520 row dataset loaded from Parquet.

The dataset has 15 columns: `marketplace`, `customer_id`, `review_id`, `product_id`, `product_parent`, `product_title`, `product_category`, `star_rating`, `helpful_votes`, `total_votes`, `vine`, `verified_purchase`, `review_headline`, `review_body`, and `review_date`.

`star_rating` has a mean of 4.17. `helpful_votes` is extremely right-skewed (mean 1.91, standard deviation 21.63, maximum 47,524). `total_votes` shows the same pattern (mean 2.58, standard deviation 23.56, maximum 48,362).

```python
df.describe("star_rating", "helpful_votes", "total_votes").show()
```

Out of the 109,830,520 total rows, 104,582,187 `review_id`s are unique, leaving 5,248,333 duplicate review IDs (~4.8%).

```python
total = df.count()
unique_reviews = df.select("review_id").distinct().count()
```

Missing values are small relative to the dataset. There are 12,438 null values for the `review_body` column, which makes up less than 0.012% of the rows. 

```python
df.select([spark_sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]).show()
```

| Column | Missing |
|---|---|
| `product_category` | 1,753 | 
| `star_rating` | 1,787 | 
| `helpful_votes` | 1,794 | 
| `total_votes` | 1,794 | 
| `vine` | 1,794 |
| `verified_purchase` | 1,794 |
| `review_headline` | 2,044 |
| `review_body` | 12,438 |
| `review_date` | 8,243 |
|(all others) | 0 |

During data exploration, we found corrupted records where dates and free-text appear in place of valid values. For example, `product_category` contained entries such as `"2002-08-07"` and `star_rating` contained date strings such as `"2011-01-24"`. The `product_category` column showed 84 distinct values (rather than the expected category set) because of these corrupted rows. Both issues are handled in preprocessing.

Our target is `helpfulness_ratio`, which is defined as `helpfulness_ratio = helpful_votes / total_votes`. It is computed only for rows where `total_votes > 0` to avoid division by zero. For modeling, this is binarized into a `label` (1.0 if ratio ≥ 0.5, else 0.0).

### 3.2 Preprocessing

Preprocessing was organized into four stages and was implemented entirely with Spark DataFrame operations and Spark MLlib transformers. The cleaned dataset is written to Parquet and checkpointed so downstream training does not rerun preprocessing.

Stage 1: Data Cleaning

```python
def filter_invalid_categories(df):
    return df.filter(col("product_category").isin(valid_categories))

def filter_invalid_star_ratings(df):
    return df.withColumn("star_rating", col("star_rating").cast("int")) \
             .withColumn("helpful_votes", col("helpful_votes").cast("int")) \
             .withColumn("total_votes", col("total_votes").cast("int")) \
             .filter(col("star_rating").isin([1, 2, 3, 4, 5]))

def drop_null_records(df):
    required = ["review_id", "review_body", "review_headline", "review_date", "product_category"]
    return df.dropna(subset=required)

def drop_duplicate_reviews(df):
    return df.dropDuplicates(["review_id"])

def impute_missing_values(df):
    imputer = Imputer(strategy="median",
        inputCols=["star_rating", "helpful_votes", "total_votes"],
        outputCols=["star_rating", "helpful_votes", "total_votes"])
    return imputer.fit(df).transform(df)
```

`product_category` was filtered to a predefined list of valid category names to remove date/text corruption. `star_rating`, `helpful_votes`, and `total_votes` were cast to integers, and rows with star ratings outside 1–5 were dropped. Rows with nulls in essential columns (`review_id`, `review_body`, `review_headline`, `review_date`, `product_category`) were dropped, duplicates were removed on `review_id` to prevent leakage between splits, and a median `Imputer` handled any remaining missing numeric values.

Stage 2: Encoding

```python
def cleansing2(df):
    df = df.withColumn("star_rating", col("star_rating").cast("int"))
    df = df.withColumn("review_headline_length", length(col("review_headline")))
    indexer_vp = StringIndexer(inputCol="verified_purchase", outputCol="verified_purchase_idx", handleInvalid="keep")
    df = indexer_vp.fit(df).transform(df)
    category_indexer = StringIndexer(inputCol="product_category", outputCol="category_idx", handleInvalid="keep")
    df = category_indexer.fit(df).transform(df)
    return df
```

`review_headline_length` was created with the `length()` function, and `StringIndexer` converted `verified_purchase` and `product_category` into numeric indices for MLlib.

Stage 3: Feature Engineering and Scaling

```python
def cleansing3(df):
    df = df.withColumn("review_length", length(col("review_body")))    
    tokenizer = Tokenizer(inputCol="review_body", outputCol="review_tokens")
    df = tokenizer.transform(df)
    return df
```

`review_length` captures the character count of the review body as a proxy for review depth. A `Tokenizer` produced `review_tokens` to enable possible text-based expansion. The logistic regression baseline applies `StandardScaler` within its own pipeline.

Stage 4: Target Engineering and Sampling

```python
def cleansing4(df):
    df = df.filter(col("total_votes") > 0)
    df = df.withColumn("helpfulness_ratio", col("helpful_votes") / col("total_votes"))
    df = df.withColumn("label", when(col("helpful_votes") / col("total_votes") >= 0.5, 1.0).otherwise(0.0))
    df = df.withColumn("log_review_length", log1p(col("review_length")))
    fractions = {1: 0.6, 2: 1.0, 3: 0.67, 4: 0.32, 5: 0.09}
    df = df.sampleBy("star_rating", fractions=fractions, seed=42)
    return df
```

Rows with `total_votes == 0` were dropped, the `helpfulness_ratio` and binary `label` were computed, and `log1p` transforms compressed the long right tails of the review length columns. Stratified sampling via `sampleBy` on `star_rating` addressed the heavy skew toward 5-star reviews.

Preprocessing reduced the data from 109,830,520 to 11,405,130 rows (an 89.62% reduction), which was driven by the removal of invalid categories, invalid star ratings, nulls, and duplicates, removal of zero-`total_votes` rows, and stratified downsampling. The resulting label distribution is 3,330,981 (label 0) and 8,074,149 (label 1). Label 1 is about 2.4 times more frequent, which motivates the use of AUC and F1 rather than accuracy.

### 3.3 Model 1 — Distributed Random Forest 

Five models were trained on a 70/15/15 train/validation/test split with `seed=42`. The five features were `star_rating`, `log_review_length`, `review_headline_length`, `verified_purchase_idx`, and `category_idx`, assembled with `VectorAssembler`.

```python
feature_cols = ["star_rating", "log_review_length", "review_headline_length",
                "verified_purchase_idx", "category_idx"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
train_df, val_df, test_df = df_clean.randomSplit([0.7, 0.15, 0.15], seed=42)
 
rf_model1 = RandomForestClassifier(featuresCol="features", labelCol="label",
                                   numTrees=20, maxDepth=15, maxBins=64, seed=42)
model_rf_model1 = Pipeline(stages=[assembler, rf_model1]).fit(train_df)
```

A logistic regression baseline (`maxIter=20`, `regParam=0.01`, with `StandardScaler`) was followed by four Random Forest configurations (`RandomForestClassifier`, all with `maxBins=64`, `seed=42`):


| Model | numTrees | maxDepth | Notes |
|---|---|---|---|
| Logistic Regression | — | — | baseline, StandardScaler |
| RF 1 | 20 | 15 | — |
| RF 2 | 20 | 5 | shallow |
| RF 3 | 50 | 15 | more trees |
| RF 4 | 30 | 12 | `featureSubsetStrategy="sqrt"` |

### 3.4 Model 2 — Dimensionality Reduction (PCA/SVD) + Downstream Analysis 

.......... The Model 2 code is in progress. 

The second model applies unsupervised dimensionality reduction (PCA) to the same engineered features, followed by a supervised Logistic Regression classifier, to test whether a compressed representation preserves predictive signal. The five features were assembled, standardized (`withMean=True, withStd=True`), reduced to k = 3 principal components with `pyspark.ml.feature.PCA`, and passed to Logistic Regression in a single Spark `Pipeline`. The same 70/15/15 split (`seed=42`) was used.
......

### 3.5 Summary of Models
 
| | Model 1  | Model 2  |
|---|---|---|
| Type | Supervised ensemble | Unsupervised reduction + supervised |
| Technique | Random Forest (and Logistic Regression baseline) | ... |
| Input features | 5 structured | ... |
| Split | 70/15/15, seed 42 | ... |
| Metrics | AUC, F1 | ... |


---

## 4. Results

Results mirror the Methods subsections. Interpretations will be mentioned in the Discussion section.

### 4.1 Data Exploration Results

Star rating distribution

<img width="500" height="335" alt="Star Rating Distribution" src="https://github.com/user-attachments/assets/ef24ccf1-e6f5-49c6-8a36-9541f780a140" />
 
*Figure 2. Distribution of star ratings across all 109M reviews. The distribution is heavily skewed toward 5 stars (~67.1M), with 1-star reviews second (~9.4M), producing a J-shaped pattern.*
 
Top 10 product categories by review count

<img width="500" height="320" alt="Top 10 Categories" src="https://github.com/user-attachments/assets/894f9df7-cbd3-4a41-8dcb-da8c4223656d" />
 
*Figure 3. The ten most-reviewed categories. Wireless leads (~9M), followed by PC and Mobile Apps. Review counts are uneven across categories.*
 
Helpfulness ratio: verified vs. unverified purchases

<img width="500" height="400" alt="Verified vs Unverified" src="https://github.com/user-attachments/assets/4822d2f4-676a-43de-94f6-d01f8f031ac1" />
 
*Figure 4. Verified purchasers show a slightly higher helpfulness ratio (Y = 0.75) than unverified (N = 0.72).*
 
Helpfulness ratio by star rating

<img width="500" height="350" alt="Helpfulness by Star Rating" src="https://github.com/user-attachments/assets/b61a473a-6002-4c31-8358-7c1e117f34a3" />

*Figure 5. Higher star ratings correspond to higher helpfulness ratios (5 stars ≈ 0.82, 1 star ≈ 0.56).*

Review length vs. helpfulness ratio

<img width="500" height="280" alt="Length vs Helpfulness" src="https://github.com/user-attachments/assets/93a06019-87fd-4950-94a7-a93ad7f8e86c" />

*Figure 6. Longer reviews tend to receive higher helpfulness ratios; short reviews (<500 chars) cluster around 0.68–0.75, while reviews near 5,000 chars reach 0.85–0.93.*

Reviews per year and average helpfulness ratio per year

<img width="900" height="330" alt="Reviews and Helpfulness over Time" src="https://github.com/user-attachments/assets/bd4c178f-6363-4bb6-abc6-7c57ca30b75d" />

*Figure 7. Review volume rises from under 5M/year before 2011 to ~30M/year by 2015, while the average helpfulness ratio declines over the same period.*

Vine vs. non-Vine helpfulness ratio

<img width="500" height="435" alt="Vine vs Non-Vine" src="https://github.com/user-attachments/assets/e0794667-5546-4602-9cdd-a5987f73abaf" />

*Figure 8. Non-Vine reviewers show a slightly higher average helpfulness ratio (N = 0.6761) than Vine reviewers (Y = 0.6412).*

### 4.2 Preprocessing Results
<img width="480" height="400" alt="Row Count Before vs  After Preprocessing" src="https://github.com/user-attachments/assets/17517888-5911-4159-a821-0313ffcb0084" />

*Figure 9. Row count before and after preprocessing. Cleaning, the `total_votes > 0` filter, deduplication, and stratified sampling reduce 109,830,520 rows to 11,405,130 (an 89.62% reduction).*
 
<img width="480" height="400" alt="Class Distribution After Preprocessing" src="https://github.com/user-attachments/assets/0ba8e380-02a2-4442-8d31-9d077d061269" />

*Figure 10. Class distribution of the binary label after preprocessing. Label 1 (helpful) totals 8,074,149 vs. 3,330,981 for label 0. There's around a 2.4:1 imbalance that motivates AUC and F1 as evaluation metrics rather than accuracy.*

### 4.3 Model 1 Results

AUC (area under the ROC curve) and F1 were used as primary metrics because the labels are imbalanced (label 1 ≈ 2.4× label 0), under which raw accuracy would be misleading.
 
| Model | Train AUC | Val AUC | Test AUC | Train F1 | Val F1 | Test F1 |
|---|---|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.6623 | 0.6628 | 0.6633 | 0.6364 | 0.6373 | 0.6360 |
| RF (numTrees=20, maxDepth=15) | 0.7096 | 0.7060 | 0.7065 | 0.6871 | 0.6854 | 0.6842 |
| RF (numTrees=20, maxDepth=5) | 0.6725 | 0.6722 | 0.6732 | 0.6398 | 0.6402 | 0.6391 |
| RF (numTrees=50, maxDepth=15) | 0.7098 | 0.7061 | 0.7068 | 0.6874 | 0.6856 | 0.6845 |
| RF (numTrees=30, maxDepth=12, sqrt) | 0.7025 | 0.7019 | 0.7027 | 0.6828 | 0.6830 | 0.6819 |
 
Table 1. Model 1 performance across train, validation, and test splits.
 
<img width="450" height="250" alt="Model Comparison Test AUC and Test F1" src="https://github.com/user-attachments/assets/bc5a0501-7a40-4fed-ac1d-bca6ab50e6f5" />

*Figure 11. Test AUC and Test F1 for all five Model 1 configurations. The deep Random Forests (maxDepth=15) lead on both metrics, while the linear baseline and the shallow forest (maxDepth=5) trail.*


Feature importance (RF numTrees=20, maxDepth=15), ranked by Gini importance from the trained model:
 
| Feature | Importance |
|---|---|
| `log_review_length` | 0.4085 |
| `category_idx` | 0.2863 |
| `star_rating` | 0.2453 |
| `review_headline_length` | 0.0443 |
| `verified_purchase_idx` | 0.0156 |
 
Table 2. Review length is the dominant predictor, followed by product category and star rating; verified-purchase status contributes least.

<img width="450" height="250" alt="Feature Importances" src="https://github.com/user-attachments/assets/0cfeffd7-9de3-478c-82a9-656745b45ddf" />

*Figure 12. Feature importances for RF (numTrees=20, maxDepth=15). `log_review_length` (0.41) dominates, followed by `category_idx` (0.29) and `star_rating` (0.25).*
 
Example ground truth vs. predictions: 

<img width="288" height="428" alt="Example predictions" src="https://github.com/user-attachments/assets/c848547b-618c-4b7e-8934-2a0add37d66d" />
 
*Figure 13. Example predictions from RF (numTrees=20, maxDepth=15) on train/validation/test, showing true label, predicted label, and the probability vector.*

### 4.4 Model 2 Results

.............

### 4.5 Speedup Analysis (Distributed Computing)
 
The representative operation timed was the full train → transform → count pipeline for RF (numTrees=20, maxDepth=15), run with 1 executor versus 15 executors on identical checkpointed data.
 
| Executors | Time (sec) | Speedup | Efficiency |
|---|---|---|---|
| 1 | 711.65 | 1.00x | 100% |
| 15 | 396.39 | 1.80x | 12.0% |
 
Table 9. Measured distributed speedup.
 
$$S = \frac{T_1}{T_{15}} = \frac{711.65}{396.39} \approx 1.80\times \qquad E = \frac{S}{n} = \frac{1.80}{15} \approx 12.0\%$$
 
$$p = \frac{n(S-1)}{S(n-1)} = \frac{15 \times 0.80}{1.80 \times 14} \approx 0.48 \qquad S_{\max} = \frac{1}{1-p} \approx 1.92\times \qquad \frac{S}{S_{\max}} = \frac{1.80}{1.92} \approx 0.94$$
 
The measured 1.80x reaches ~94% of the theoretical Amdahl limit of 1.92x, so the implementation is near its practical scaling ceiling. The ~48% parallelizable fraction reflects sequential dependencies in tree growth, driver-side coordination, and shuffle costs that do not shrink as executors are added.
 
---

## 5. Fitting Analysis
 
<img width="500" height="300" alt="Fitting Graph" src="https://github.com/user-attachments/assets/02f94320-e56e-4008-9623-4de6e802a57d" />


*Figure 17. Fitting graph: Train and Test AUC as model capacity increases from the linear baseline through deeper Random Forests. Train and Test curves track each other closely at every capacity level, indicating no significant overfitting; performance rises with capacity and plateaus at maxDepth=15.*
 
### Where each model fits on the fitting graph
 
The logistic regression baseline and the shallow Random Forest (maxDepth=5) sit in the underfitting region: their train and test AUCs are both low (~0.66–0.67) and nearly identical, the signature of insufficient model capacity rather than overfitting. The deeper Random Forests (maxDepth=12 and 15) sit in the good-fit region, with train/test AUC gaps under 0.004 and the highest absolute scores (~0.71). Performance plateaus between 20 and 50 trees at maxDepth=15, indicating the forest has converged.
 
### Where Model 2 (PCA + LR) fits on the fitting graph
 
...
 
### Future improvements and next models
 
- Apply dimensionality reduction to high-dimensional inputs. PCA and SVD are most beneficial on large, correlated feature spaces such as TF-IDF or Word2Vec vectors from `review_body`, where compression removes genuine redundancy. The `Tokenizer` already in the pipeline makes this a natural next step.
- SVD / LSA on text features. Factorizing a TF-IDF matrix of the review body into latent semantic components could surface structure invisible in the five structured features.
- Nonlinear classifiers (Gradient-Boosted Trees) on the reduced or full feature space would capture interactions a linear model cannot. .......

### How dimensionality reduction affected results vs. the full feature set
 
Reducing from five features to three components (70.9% variance retained) lowered performance at every stage. Test AUC fell from 0.6633 (full-feature LR) and 0.7065 (full-feature RF) to 0.6347, which is an AUC gap of 0.0718 versus the best Random Forest. PC1 cleanly absorbs the three correlated "effort" features, but `star_rating` and `category_idx` carry largely independent variance spread across PC2 and PC3, so truncation removes discriminative signal rather than noise. .....
 
---
 
## 6. Discussion
 
This section presents our interpretation and reasoning across both models, and where we are skeptical of our own results.
 
Data exploration: The exploratory findings are internally consistent and based on the full 109M rows, which makes the aggregate trends trustworthy. The J-shaped star distribution matches well-documented review behavior (people review when very satisfied or very dissatisfied). The positive associations between review length and helpfulness, and between higher star ratings and helpfulness, both pointed toward features worth engineering. We are appropriately cautious about the verified-vs-unverified and Vine-vs-non-Vine gaps: although computed over very large samples, the differences are small (~0.03), and the two comparisons used slightly different aggregation methods, so they should be read as suggestive rather than definitive.
 
Preprocessing: The 89.62% row reduction is large and deserves scrutiny. Most of it is expected and defensible: dropping reviews with `total_votes == 0` removes the large majority of rows (most reviews are never voted on), and stratified downsampling deliberately discards a large share of 5-star reviews to balance the classes. A shortcoming worth noting is that aggressive filtering plus downsampling means our models describe the *voted, class-balanced* subset of reviews rather than the full population, which limits how far conclusions generalize to never-voted reviews.
 
Model 1: The best Random Forest reaches a test AUC of ~0.71, a clear improvement over the logistic-regression baseline (0.6633), and the train/test AUC gap under 0.004 indicates the model generalizes rather than memorizes. We interpret the logistic regression and shallow (maxDepth=5) forest as underfitting because low scores with near-zero gaps point to insufficient capacity, implying a nonlinear relationship between these features and helpfulness. The most informative result is the feature importance. `log_review_length` dominates (0.41), followed by `category_idx` (0.29) and `star_rating` (0.25). This aligns with the exploratory finding that longer reviews are rated more helpful and suggests effort/detail matters more than the rating itself. 
 
Model 2: ..........
 
........ This is where you discuss the "why" and your interpretation—your thought process from beginning to end. Discuss how believable your results are at each step. Discuss any shortcomings. It's okay to criticize your own work—this shows intellectual merit and scientific thinking. In science we rarely find perfect solutions. If your results seem too good, scrutinize them carefully! .....
 
---
 
## 7. Conclusion
 
### Conclusion of Model 1
 
Using five structured metadata features, the best model was the Random Forest with numTrees=20 and maxDepth=15. It achieved a test AUC of 0.7065 and test F1 of 0.6842, improving on the logistic baseline by 0.043 AUC. RF (numTrees=50, maxDepth=15) scored marginally higher (test AUC 0.7068) but at substantially higher training cost for a +0.0003 gain, so the 20-tree model is preferred for its efficiency/quality trade-off on the full 50 GB dataset. The consistent sub-0.004 train/val/test gap confirms good generalization, while the ~0.71 ceiling shows structured metadata alone provides a meaningful but limited signal.
 
To improve Model 1, we can incorporate text features from `review_body` (TF-IDF or Word2Vec on the already-created tokens), which likely carry the strongest helpfulness signal and experiment with class weights or revised sampling fractions to lift minority-class F1.
 
### Conclusion of Model 2
 
.........
 
### What We Learned
This project demonstrated that distributed computing is not optional at this scale. We learned how Spark partitions data and parallelizes both aggregations and ensemble training, how checkpointing avoids recomputing expensive preprocessing, and why metric choice (AUC/F1 over accuracy) matters under class imbalance. Distributed computing changed our approach by letting us iterate across many model configurations within a single session, which would have been impractical on a single core. We also learned that more executors do not guarantee proportional speedup, and PCA does not automatically always help. With more time and resources, we would add review-text features and extend the dimensionality-reduction experiments to that larger feature space, where PCA and SVD are far more likely to provide real benefit, and we would study scaling behavior beyond 15 executors.....
 
---

## 8. Statement of Collaboration

Kayla Zhu: Writer/ Coder: Wrote the project abstract; contributed Milestone 2 code (data exploration and visualizations); wrote the Milestone 2 README; completed the Milestone 3 code (preprocessing pipeline, trained and compared the logistic regression baseline and eight Random Forest configurations, example ground-truth and predictions, fitting analysis, hyperparameter comparison, visualizations, and speedup analysis); wrote the Milestone 3 README; co-wrote the final Milestone 4 README; provided feedback throughout.

Name: Title: Contribution

Name: Title: Contribution

Name: Title: Contribution

---

## 9. Notebooks

All code is provided as Jupyter notebooks that can be followed in order. Outputs and inline visualizations are pre-rendered.

- Data Exploration (Milestone 2): [Milestone2.ipynb](Milestone2.ipynb)
- Preprocessing & First Distributed Model (Milestone 3): [Milestone3.ipynb](Milestone3.ipynb)
- Second Model Using Dimensionality Reduction (Milestone 4): [Milestone4.ipynb](Milestone4.ipynb)
