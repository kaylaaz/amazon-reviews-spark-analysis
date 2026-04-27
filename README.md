# Amazon US Customer Reviews Spark Analysis

## Dataset
**Source:** [Amazon US Customer Reviews Dataset](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset)

The Amazon US Customer Reviews Dataset is a collection of product reviews submitted. It contains 109,830,520 total rows across 37 TSV files, spanning 37 product categories from 1995 to 2015. The total dataset size is approximately 50.68 GB, which makes it well-suited for processing with Spark. 

Each record represents a single customer review and includes both structured metadata and unstructured text. This project uses these features to analyze and predict review helpfulness.

## SDSC Expanse Environment Setup
This project was developed and executed on the San Diego Supercomputer Center (SDSC) Expanse cluster.

Given the dataset size, significant RAM and parallelism are required. The configuration below is designed to handle both volume and memory.


| Parameter | Value | 
| --- | --- |
| Partition | shared | 
| Type | JupyterLab |
| Total Cores | 16 | 
| Total Memory | 128 GB | 
| Driver Cores | 1 | 
| Executor Instances | 15 | 
| Executor Memory | 8.4 GB | 
| Total Memory Used | 122 GB | 

**Formulas used:**

`Executor instances = Total Cores - 1` = 16 - 1 = 15

`Executor memory = (Total Memory - Driver Memory) / Executor Instances` = (128 - 2) / 15 = 8.4 GB

`Total Memory Used` = 2 + (8 x 15) = 122 GB

**Justification:** 128 GB total memory with 16 cores provides the right balance for this workload. The 50.68 GB dataset expands significantly in memory during Spark operations. Allocating 8 GB per executor prevents out-of-memory errors during aggregation-heavy operations and Word2Vec model training. Reserving one core and 2 GB for the driver ensures cluster stability while maximizing parallel task execution across 15 worker executors.



### SparkSession Configuration
```
spark = SparkSession.builder \
    .config("spark.driver.memory","2g") \
    .config("spark.executor.memory", "8g") \
    .config('spark.executor.instances', 15) \
    .config("spark.executor.cores", "1") \
    .getOrCreate()
```

### Spark UI Screenshot
<img width="500" height="350" alt="Screenshot of Spark UI" src="https://github.com/user-attachments/assets/2b9e941a-624a-41d4-bb1b-07a77359f259" />

This screenshot was taken during the initial data loading phase. We can see multiple executors of the Spark UI work from Expanse, and the code runs simultaneously. 

## Data Exploration using Spark
All exploration is performed using Spark DataFrames. 

### How Many Observations Does Your Dataset Have? 
The full dataset contains 109,830,520 rows and 15 columns loaded from 37 TSV files spanning 20 years (1995 to 2015).

### Column Descriptions 

#### Categorical Variables 

| Column Name | Type | Scale | Distribution / Description | 
| --- | --- | --- | --- |
| `marketplace` | string | Nominal | Only 1 unique value (`US`) across all 109 million rows. No predictive variance. |
| `product_category` | string | Nominal | 84 unique values found despite 37 files. Categories include Wireless, PC, Mobile_Apps, Books, Electronics, and more. Wireless is the largest with ~9M reviews. Note: 34 corrupted records contain dates instead of category names (e.g., "2002-08-07"). |
| `vine` | string | Nominal | Binary (`Y` or `N`). Has a small number of nulls. Indicates Amazon Vine program membership. |
| `verified_purchase` | string | Nominal | Binary (`Y` or `N`). The majority of reviews are from verified purchasers. Small number of nulls. |
| `star_rating` | string | Ordinal (1–5) | Stored as string. Distribution is heavily skewed toward 5 stars (~67M of 109M rows). Note: Contains corrupted records where dates appear instead of ratings (e.g., "2011-01-24"). |
| `review_id` | string | Nominal | Unique identifier per review. Can be used for duplicate detection. |
| `product_id` | string | Nominal | Unique identifier for each product. |
| `customer_id` | integer | Nominal | Customer identifier. High cardinality. |
| `product_parent` | integer | Nominal | Groups product variants under a common parent. |

#### Continuous Variables
| Column Name | Type | Scale | Distribution / Description |
|---|---|---|---|
| `helpful_votes` | integer | Ratio | Number of helpful votes a review received. Extremely right-skewed (most reviews receive very few helpful votes). Mean = 1.91, Std = 21.63, Max = 47,524. |
| `total_votes` | integer | Ratio | Total number of votes a review receieved. Same right-skewed distribution as `helpful_votes`. Mean = 2.58, Std = 23.56, Max = 48,362.  |

#### Text Variables
| Column Name | Type | Description |
|---|---|---|
| `review_headline` | string | Short title of the review. Variable length. |
| `review_body` | string | Full text of the review. Has the most missing values (12,438 nulls).  |
| `review_date` | date | Interval scale. Ranges from 1995 to 2015. |
| `product_title` | string | Product name. Free text. High cardinality. |

#### Target Variable
Our target variable is `helpfulness_ratio`, which is computed as: `helpfulness_ratio = helpful_votes / total_votes`

This ratio represents how useful other shoppers found a given review, ranging from 0.0 (no one found it helpful) to 1.0 (everyone found it helpful). Only rows where `total_votes > 0` are included in this computation to prevent division by zero. This frames our prediction task as a regression problem on a bounded continuous target. 

### Missing Values
| Column Name | Missing Values |
|---|---|
| `marketplace` | 0 |
| `customer_id` | 0 |
| `review_id` | 0 |
| `product_id` | 0 |
| `product_parent` | 0 |
| `product_title` | 0 |
| `product_category` | 1,753 |
| `star_rating` | 1,787 |
| `helpful_votes` | 1,794 |
| `total_votes` | 1,794 |
| `vine` | 1,794 |
| `verified_purchase` | 1,794 |
| `review_headline` | 2,044 |
| `review_body` | 12,438 |
| `review_date` | 8,243 |

The amount of missing values is considered small relative to the 109 million rows in the dataset. The largest gap is `review_body` (12,438 nulls), and it represents less than 0.012% of the dataset. Based on the data, the missing values don't seem to be a major concern, and they will be handled during preprocessing. 

### Duplicate Values
| Metric | Value |
|---|---|
| Total Rows | 109,830,520 |
| Unique `review_id`s | 104,582,187 |
| Duplicate `review_id`s | 5,248,333 (~4.8% of the dataset)|

Approximately 4.8% of the `review_id`s are duplicates or resubmitted reviews. These will be addressed by deduplicating on `review_id` during preprocessing to prevent skewed model training. 

## Data Plots

### Plot 1 - Star Rating Distribution (Bar Chart)
<img width="500" height="335" alt="Plot 1" src="https://github.com/user-attachments/assets/ef24ccf1-e6f5-49c6-8a36-9541f780a140" />

This bar chart shows the distribution of star ratings across all 109 million reviews. The distribution is heavily skewed toward 5-star ratings, which account for approximately 67 million reviews. 1-star ratings are the second most common at ~9.4 million. A J-shaped distribution is produced, which could mean that people are motivated to review when they are either very satisfied or very disappointed. This class imbalance will need to be addressed during preprocessing. 

### Plot 2 - Top 10 Product Categories by Review Count (Horizontal Bar Chart)
<img width="500" height="320" alt="Plot 2" src="https://github.com/user-attachments/assets/894f9df7-cbd3-4a41-8dcb-da8c4223656d" />

This horizontal bar chart shows the 10 most reviewed product categories. Wireless leads with ~9 million reviews, followed by PC and Mobile Apps. The distribution across categories is uneven, which means our analysis must account for this imbalance when comparing helpfulness metrics across categories. 

### Plot 3 - Helpfulness Ratio: Verified vs. Unverified Purchases (Bar Chart)
<img width="500" height="400" alt="Plot 3" src="https://github.com/user-attachments/assets/4822d2f4-676a-43de-94f6-d01f8f031ac1" />

This bar chart compares the average helpfulness ratio between verified (Y = 0.75) and unverified (N = 0.72) purchasers. Verified purchasers have a slightly higher helpfulness ratio, suggesting that shoppers marginally trust reviews from people who actually bought the product. The difference is small but consistent across 109 million rows, making it statistically meaningful.

### Plot 4 - Helpfulness Ratio by Star Rating (Bar Chart)
<img width="500" height="350" alt="Plot 4" src="https://github.com/user-attachments/assets/b61a473a-6002-4c31-8358-7c1e117f34a3" />
 
This bar chart shows a clear positive trend where higher star ratings correspond to higher helpfulness ratios. 5-star reviews have a helpfulness ratio of ~0.82, while 1-star reviews have the lowest at ~0.56. This suggests positive reviews are generally found more useful by other shoppers. 

### Plot 5 - Review Length vs. Helpfulness Ratio (Scatter Plot)
<img width="500" height="280" alt="Plot 5" src="https://github.com/user-attachments/assets/93a06019-87fd-4950-94a7-a93ad7f8e86c" />

This scatter plot shows the relationship between review length (in characters) and helpfulness ratio, sampled from reviews with at least 1 vote. A clear positive trend is visible, which implies that longer reviews tend to receive higher helpfulness ratios. Reviews under 500 characters cluster around 0.68–0.75, while reviews approaching 5,000 characters reach ratios of 0.85–0.93. This supports our hypothesis that review length is a meaningful predictor of perceived helpfulness.

### Plot 6 - Number of Reviews vs. Average Helpfulness Ratio Per Year (Line Chart) 
<img width="900" height="330" alt="Plot 6" src="https://github.com/user-attachments/assets/bd4c178f-6363-4bb6-abc6-7c57ca30b75d" />

These line chart plots review volume and average helpfulness ratio over time. A gradual decline in the average helpfulness ratio is visible even as review volume increases. The number of reviews per year increases from just under 5 million reviews per year before 2011 to approximately 30 million reviews by 2015. Earlier years show higher helpfulness scores, which may be due to a smaller and more engaged user base. As Amazon scaled, rapid growth introduced more low-effort, short, or spam reviews that tend to receive fewer helpful votes, lowering the platform-wide average. 

### Plot 7 - Average Helpfulness Ratio for Vine vs Non-Vine (Bar Chart) 
<img width="500" height="435" alt="Plot 7" src="https://github.com/user-attachments/assets/e0794667-5546-4602-9cdd-a5987f73abaf" />

This bar chart compares the average helpfulness ratio between Vine (Y = 0.6412) and non-Vine (N = 0.6761) reviewers. Amazon Vine is an invitation-only program where members receive free products in exchange for reviews. From the chart, we can see that non-Vine reviewers score slightly higher. A likely explanation is that shoppers inherently trust reviews from people who paid for the product more than those who received it for free, perceiving the latter as potentially biased. This suggests the `vine` column could have some predictive value.

## Preprocessing Plan
This section describes the planned preprocessing approach. Implementation will be completed in Milestone 3. 

### Handling Missing Values
Missing values represent less than 0.01% of the 109 million row dataset, so dropping affected rows will not meaningfully reduce data quality. We plan to drop rows where critical columns are null using `df.dropna(subset=[...])`, specifically targeting: `star_rating`, `review_body`, `helpful_votes`, and `total_votes`.

### Handling Data Imbalance
Star ratings are heavily skewed toward 5 stars (around 67 million out of 109 million rows). The imbalance could cause models to be biased toward predicting positive reviews. We plan to use stratified sampling via `df.sampleBy()` to balance star rating classes during model training. 

### Cleaning Data
During exploration, we found that `product_category` and `star_rating` had some data quality issues that must be resolved before modeling. They can be filtered out using `df.filter()`.

`product_category` contains 34 records where full review text or dates appear instead of a category name (e.g., "2002-08-07"). The review text and dates will be filtered out.

`star_rating` contains records where dates appear instead of ratings (e.g., "2011-01-24"). These values will be filtered to only valid values (1 through 5).

### Transformations
Some transformations that we'll apply are: 
- Cast `star_rating` from string to integer using `df.withColumn()` and `.cast('int')`
-  Remove corrupted `star_rating` values (dates were found during data exploration) by filtering to only valid values (1, 2, 3, 4, or 5)
- Compute `review_length` as the character count of `review_body` using `length(col('review_body'))`
- Compute `helpfulness_ratio = helpful_votes / total_votes` as the target variable. This is only applied to rows where `total_votes > 0` to prevent division by zero
- Encode `verified_purchase` and `vine` (currently either Y or N) to binary 0 or 1 using `StringIndexer` so they can be passed into MLlib models as numeric features


### Spark Operations for Preprocessing
Some Spark operations that we'll use for preprocessing include: 
- `df.dropna()` to drop rows with null values in critical columns (e.g., `star_rating`, `review_body`, `helpful_votes`, `total_votes`)
- `df.withColumn()` for feature engineering steps like casting, computing `review_length`, and deriving the `helpfulness_ratio`
- `df.filter()` to remove corrupted rows (e.g., invalid values, zero `total_votes`)
- `df.sampleBy()` to perform stratified sampling to balance some skewed class distributions 
- `MLlib StringIndexer` to encode categorical columns into numeric indices 


## Notebook
**Jupyter notebook:** [Milestone2.ipynb](Milestone2.ipynb)

All data exploration, aggregations, and plot generation code is contained in the notebook above. Outputs and inline visualizations are pre-rendered for review. 
