# Amazon US Customer Reviews Spark Analysis
This project uses the [Amazon US Customer Reviews](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset) dataset sourced from Kaggle. The dataset is approximately 50.68 GB and contains over 109 million rows spanning dozens of product categories. The goal is to predict whether a review is helpful, defined as having a helpfulness ratio (`helpful_votes` / `total_votes`) of at least 0.5.


## 1. Preprocessing
All preprocessing was implemented using Spark DataFrame operations and Spark MLlib transformers on SDSC Expanse. The preprocessing pipeline is divided into four stages and is fully reproducible by running the notebook cells in order. The cleaned dataset is saved to Parquet and checkpointed so downstream model training does not need to rerun preprocessing.

### Step 1: Data Cleaning
The `product_category` column contained invalid entries, such as dates and free-form review text. These were removed by filtering to a list of known valid categories. The `star_rating`, `helpful_votes`, and `total_votes` columns were cast from strings to integers, and rows with star ratings outside the valid range of 1 to 5 were dropped. Rows with null values in `review_id`, `review_body`, `review_headline`, `review_date`, or `product_category` were dropped because these fields are essential for both analysis and modeling. Duplicate reviews were removed by deduplicating on `review_id` to prevent data leakage between train and test splits. `Imputer` from `pyspark.ml.feature` was then applied to `star_rating`, `helpful_votes`, and `total_votes` using a median strategy to handle any remaining missing numeric values.

### Step 2: Encoding
`star_rating` was cast to an integer to ensure the correct numeric type after imputation. A new column `review_headline_length` was created by applying the `length()` Spark SQL function to the `review_headline` column. `StringIndexer` was applied to `verified_purchase` to produce `verified_purchase_idx` and to `product_category` to produce `category_idx`, converting categorical string columns into numeric indices compatible with MLlib model inputs.

### Step 3: Feature Engineering and Scaling
`review_length` was computed by applying the `length()` Spark SQL function to the `review_body` column, capturing the total character count of the review text as a proxy for review depth and quality. A `Tokenizer` was applied to `review_body` to produce `review_tokens`, which splits the review text into individual words. The logistic regression baseline applies `StandardScaler` through its own pipeline stage.

### Step 4: Target Engineering and Sampling
Rows where `total_votes` equals 0 were removed because a helpfulness ratio cannot be computed for them. The target variable `helpfulness_ratio` was computed as `helpful_votes / total_votes`. A binary label column was created where label = 1.0 if helpfulness_ratio >= 0.5 and label = 0.0 otherwise. Log-transformed columns were created using the `log1p` Spark SQL function. `log_review_length` is used as a model feature. The `log1p` transformation compresses the long right tail in review lengths. Stratified sampling was applied using `sampleBy` on `star_rating` with fractions {1: 0.6, 2: 1.0, 3: 0.67, 4: 0.32, 5: 0.09} to address the heavy skew toward 5-star reviews in the original dataset.

## 2. First Distributed Model
### Setup
All models were trained on SDSC Expanse using a Spark session configured with 15 executors, 1 core per executor, and 8 GB of memory per executor. Multiple executors were verified and are shown in the screenshot below:
<img width="523" height="57" alt="Screenshot" src="https://github.com/user-attachments/assets/b72c09b8-759e-4806-9b4c-eaa612b78c60" />

This screenshot was taken during the initial data loading phase. Multiple active executors are visible in the Spark UI, confirming that the job ran in a distributed configuration across SDSC Expanse resources rather than locally.

The five features used for all models are `star_rating`, `log_review_length`, `review_headline_length`, `verified_purchase_idx`, and `category_idx`. These features were selected because they are available for every review, require no text processing, and cover the rating scale, review effort, purchase authenticity, and product domain. The dataset was split into train (70%), validation (15%), and test (15%) sets using a fixed random seed of 42 for reproducibility.

### Models
Five models were trained. A logistic regression model was trained as a baseline using `LogisticRegression` from `pyspark.ml.classification` with `maxIter=20` and `regParam=0.01`, preceded by a `StandardScaler` in a `Pipeline`. Four random forest (RF) models were trained using `RandomForestClassifier` from `pyspark.ml.classification` with varying hyperparameters to support the fitting analysis. All random forest models used `maxBins=64` and `seed=42`.

### Results
| Model | Train AUC | Val AUC | Test AUC | Train F1 | Val F1 | Test F1 |
|---|---|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.6623 | 0.6628 | 0.6633 | 0.6364 | 0.6373 | 0.6360 |
| RF (numTrees=20, maxDepth=15) | 0.7096 | 0.7060 | 0.7065 | 0.6871 | 0.6854 | 0.6842 |
| RF (numTrees=20, maxDepth=5) | 0.6725 | 0.6722 | 0.6732 | 0.6398 | 0.6402 | 0.6391 |
| RF (numTrees=50, maxDepth=15) | 0.7098 | 0.7061 | 0.7068 | 0.6874 | 0.6856 | 0.6845 |
| RF (numTrees=30, maxDepth=12, sqrt) | 0.7025 | 0.7019 | 0.7027 | 0.6828 | 0.6830 | 0.6819 |
 
AUC (area under the ROC curve) and F1 score were used as the primary evaluation metrics because the label distribution is imbalanced. Label 1 accounts for approximately 2.4 times as many rows as label 0. Under this imbalance, a classifier that predicts the majority class for every input would achieve high accuracy while providing no useful predictions. AUC measures the model's ability to rank helpful reviews above unhelpful ones across classification thresholds, and F1 balances precision and recall. Both metrics are more informative than accuracy for imbalanced binary classification.

### Ground Truth and Predictions
Example predictions for the first five rows from RF (numTrees=20, maxDepth=15) on the train, validation, and test sets are shown in the screenshot below. Each example displays the true label, the predicted label, and the probability vector (the model's estimated probability of label 0 and label 1).

<img width="288" height="428" alt="Example predictions" src="https://github.com/user-attachments/assets/c848547b-618c-4b7e-8934-2a0add37d66d" />

These examples confirm that the model produces probabilities and correctly classifies reviews in the majority of cases across all three splits.

### Feature Importance
The most important feature for RF (numTrees=20, maxDepth=15) is `star_rating`, followed by `log_review_length` and `review_headline_length`. `verified_purchase_idx` and `category_idx` contribute less to the model's decisions. The dominance of `star_rating` suggests that the rating a reviewer assigns is strongly predictive of whether other users find the review useful. The importance of review length indicates that longer reviews tend to be more helpful, consistent with the intuition that more detailed reviews provide more value to potential buyers.

## 3. Fitting Analysis
### Where each model fits on the fitting graph
The logistic regression model underfits. The train AUC of 0.6623 and test AUC of 0.6633 are both low, and the near-zero train/test gap confirms the issue is insufficient model capacity rather than overfitting. A linear decision boundary cannot represent the nonlinear relationships between star rating, review length, and helpfulness present in this dataset.

RF (numTrees=20, maxDepth=5) also underfits. The test AUC of 0.6732 is only marginally better than logistic regression. Trees with a maximum depth of 5 are not deep enough to capture the interactions between all five features. The model can identify broad patterns but cannot distinguish the finer combinations of feature values that separate helpful from unhelpful reviews.

RF (numTrees=20, maxDepth=15), RF (numTrees=50, maxDepth=15), and RF (numTrees=30, maxDepth=12, sqrt) all fall in the good fit region. Their train/test AUC gaps are all under 0.004, indicating that the models generalize well without memorizing the training set. The small and consistent gap across train, validation, and test splits shows that these models may have found a stable decision boundary that transfers to unseen examples.

### Hyperparameter Comparison
The most impactful hyperparameter is `maxDepth`. Increasing it from 5 to 15 raises test AUC from 0.6732 to 0.7065, a gain of 0.033. This large jump confirms that the underlying patterns in the data require deeper decision boundaries. Each additional level of depth allows the tree to condition on finer combinations of feature values, capturing interactions that shallower trees miss entirely.

Increasing `numTrees` from 20 to 50 at maxDepth=15 raises test AUC by only 0.0003, from 0.7065 to 0.7068. With only 5 features, the variance of the forest is already low at 20 trees because there are limited ways to split the feature space. Adding more trees beyond the point of convergence averages together trees that are making nearly identical decisions, so adding more provides no meaningful improvement.

Using `featureSubsetStrategy="sqrt"` with numTrees=30 and maxDepth=12 produces a test AUC of 0.7027, falling between the shallow and deep models. The square root strategy restricts each split candidate to approximately 2 of the 5 features, which introduces more diversity among trees and reduces correlation between them. With only 5 features, restricting splits to roughly 2 features removes too much information at each split.

### Best Model
The random forest (numTrees=50, maxDepth=15) achieved the highest predictive performance with a test AUC of 0.7068 and a test F1 of 0.6845. However, the improvement over RF (numTrees=20, maxDepth=15) was extremely small (+0.0003 AUC and +0.0003 F1) despite requiring substantially more training time due to the additional trees. Therefore, RF (numTrees=20, maxDepth=15) is the preferred practical model because it provides nearly identical predictive performance at a lower computational cost. On the full 50 GB dataset, training time scales with tree count, so the 20-tree model offers a better tradeoff between efficiency and model quality.
 
### Next Models for Milestone 4
PCA (Principal Component Analysis) followed by a supervised model is planned as the second model. PCA will compress the 5 structured features into a smaller number of principal components that capture the directions of maximum variance in the data. A random forest will then be retrained on the reduced feature space, allowing a direct comparison between full-feature and reduced-feature performance. This will reveal whether the original features contain redundant or correlated information that PCA can eliminate.
 
SVD (Singular Value Decomposition) is another option. SVD can factorize the feature matrix into lower-dimensional latent representations and may reveal structure not visible in the original feature space.

## 4. Conclusion
### Conclusion of the First Model
Using five structured metadata features (`star_rating`, `log_review_length`, `review_headline_length`, `verified_purchase_idx`, and `category_idx`), the best model RF (numTrees=20, maxDepth=15) achieved a test AUC of 0.7065 and a test F1 of 0.6842. This is a meaningful improvement over the logistic regression baseline of 0.6633 AUC, representing a gain of 0.043 points. The consistent train/val/test gap of less than 0.004 across all three splits confirms that the model generalizes well and is not overfitting. At the same time, a test AUC of approximately 0.71 indicates that structured metadata alone provides a limited signal for predicting helpfulness. The content of the review itself is not captured by any of the five features and likely carries substantially more predictive information.

### What Can Be Done to Improve It
The most promising improvement would be incorporating text features from the `review_body` column. Adding TF-IDF vectors or Word2Vec embeddings would give the model access to the actual words used in the review, which is likely the strongest predictor of whether other users find it helpful. Reviews that are specific, detailed, and comparative tend to be rated as more helpful, and these qualities can only be captured through the text itself. The tokenizer already applied during preprocessing makes this a natural next step. Applying PCA before training may also reduce noise from correlated features. Applying class weights during training or further adjusting the sampling fractions could improve F1 on the minority class.

### How Distributed Computing Helped
Training on SDSC Expanse with 15 executors made it practical to train and compare five models on 11 million rows within a single session. The most complex model, RF (numTrees=50, maxDepth=15), involves building 50 deep decision trees across a large partitioned dataset. Spark distributed the tree-building tasks across executors in parallel, with each executor processing a separate partition of the training data independently and contributing to the construction of each tree through aggregated split statistics. Without parallelization, iterating across five model configurations at this data scale would have required many hours on a single core, making hyperparameter comparison effectively impractical. Spark's distributed execution made it feasible to perform the full training, evaluation, and feature importance analysis for each model.


## 5. Speedup Analysis
### Methodology
The operation timed was the full training and evaluation pipeline for RF (numTrees=20, maxDepth=15): pipeline fit on the training set, transform on all three splits, and `.count()` on each split to force Spark to materialize all lazy transformations and produce an accurate wall-clock measurement. The same checkpointed dataset and identical code were used for both configurations. The 1-executor baseline and the 15-executor scaled run both used 8 GB executor memory and 1 executor core per executor.

### Results
| Executors | Time (sec) | Speedup | Efficiency |
|---|---|---|---|
| 1 | 711.65 | 1.00x | 100% |
| 15 | 396.39 | 1.80x | 12.0% |

### Calculations
Speedup:
 
$$S = \frac{T_1}{T_{15}} = \frac{711.65}{396.39} \approx 1.80\text{x}$$
 
Efficiency:
 
$$E = \frac{S}{n} = \frac{1.80}{15} \approx 12.0\%$$
 
Parallelizable fraction (Amdahl's Law):
 
$$p = \frac{n(S - 1)}{S(n - 1)} = \frac{15 \times 0.80}{1.80 \times 14} \approx 0.48$$
 
Theoretical maximum speedup:
 
$$S_{\max} = \frac{1}{1 - p} \approx 1.92\text{x}$$

Amdahl Efficiency vs Limit:

$$
\frac{S}{S_{\max}} = \frac{1.80}{1.92} = 0.9375
$$

### Analysis
The measured speedup of 1.80x reaches approximately 94% of the theoretical Amdahl limit of 1.92x, meaning our implementation is already near the practical scaling limit for this workload. The estimated parallelizable fraction of 48% indicates that adding executors provides limited returns because a substantial portion of the workload remains sequential. Key constraints on scaling include the limited parallelism available in Random Forest training, driver-side coordination overhead, and shuffle costs that do not shrink as executor count increases.

## Notebook
**Jupyter notebook:** [Milestone3.ipynb](Milestone3.ipynb)
