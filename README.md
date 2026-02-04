# Data Science Club Assignment: E-commerce Insights and Edge AI Document Triage

This repository contains two data science projects completed for the Data Science Club assignment. Task 1 focuses on deriving business insights from an e-commerce dataset, while Task 2 implements a machine learning pipeline to build an efficient, edge-first document triage system.

## 1. Problem Statement

**Task 1 (Business Intelligence):** To analyze the behavior of e-commerce customers to identify key drivers of spending and satisfaction. The goal is to extract actionable insights that can inform targeted marketing strategies, improve customer retention, and guide loyalty program enhancements.

**Task 2 (ML Engineering):** Large-scale Retrieval-Augmented Generation (RAG) systems suffer from high operational costs and latency, as they process all incoming documents regardless of quality. Passing junk data (spam, boilerplate, low-information noise) to expensive embedding models wastes significant compute cycles and energy, a problem especially detrimental for resource-constrained edge devices. The goal is to build a lightweight, CPU-efficient classification pipeline that acts as a "gatekeeper," filtering out low-quality documents before they reach costly downstream RAG stages.

## 2. Dataset Description

**Task 1: E-commerce Customer Behavior**

- **Dataset Source:** Kaggle (`uom190346a/e-commerce-customer-behavior-dataset`)
- **Number of Samples and Features:** 350 records and 11 features.
- **Target Variable (for analysis):** `Satisfaction Level`.
- **Data Types Overview:** Includes numerical data (Age, Total Spend), categorical data (Gender, City, Membership Type), and boolean data (Discount Applied).

**Task 2: Document Triage Dataset**

- **Dataset Source:** A custom-built dataset combining two real-world sources:
  - **High-Value (Label=1):** 1,000+ academic abstracts from the arXiv API (categories: `cs.AI`, `cs.LG`, `cs.OS`).
  - **Junk/Noise (Label=0):** SMS messages from the UCI SMS Spam Collection dataset.
- **Number of Samples and Features:** 2,000 balanced records with 2 features (`text`, `label`).
- **Target Variable:** `label` (1 for High-Value, 0 for Junk).

## 3. Data Cleaning & Preprocessing

**Task 1:**

- **Missing Value Handling:** (Inferred from report) Null entries in `Satisfaction_Level` were imputed with "Neutral" to prevent biasing the analysis.
- **Schema Standardization:** Column names were converted to a standardized `snake_case` format for easier programmatic access (e.g., "Total Spend" became `Total_Spend`).
- **Encoding Techniques:** The categorical `Satisfaction_Level` was ordinally encoded to a numeric scale (`Satisfaction_Numeric`) for correlation analysis.

**Task 2:**

- **Text Cleaning:** A custom `TextDistiller` transformer was built to programmatically remove boilerplate noise, including emails, URLs, and common footer text (e.g., "terms and conditions").
- **Assumptions Made:** It was assumed that academic abstracts represent high-information-density content, while SMS spam represents low-value noise, providing a clear basis for the binary classification task.

## 4. Exploratory Data Analysis (EDA)

**Task 1:**

- **Key Statistical Observations:** The mean `Total_Spend` is significantly influenced by `Membership_Type`, with Gold members spending the most. The age of customers shows very little variation and does not appear to be a primary driver of purchasing behavior.
- **Correlation Analysis:** A heatmap revealed a strong positive correlation (>0.90) between `Total_Spend`, `Items_Purchased`, and `Average_Rating`. Customer satisfaction (`Satisfaction_Numeric`) was also highly correlated with `Total_Spend` (r=0.80). Conversely, `Age` had a near-zero correlation with spending.
- **Patterns and Trends:** Spending is concentrated in specific membership tiers. The application of discounts is strongly associated with an increase in the quantity of items purchased.

## 5. Visualizations

**1. Membership Spend Distribution (Violin Plot)**

- **What it shows:** This visualization displays the distribution of `Total_Spend` across the three `Membership_Type` categories (Bronze, Silver, Gold).
- **Why it is important:** It helps visualize both the range and density of spending for each customer segment.
- **Insight:** Gold members have a much higher and more tightly clustered spending distribution (a high "spending floor" around $600-$1500), whereas Bronze and Silver tiers exhibit wider, lower-value variance. This confirms the loyalty program successfully segments high-value customers.

**2. Discount Efficacy on Purchase Volume (Box Plot)**

- **What it shows:** This box plot compares the distribution of `Items_Purchased` for transactions with and without a discount applied.
- **Why it is important:** It directly measures the impact of a key marketing lever (discounts) on customer purchasing volume.
- **Insight:** Applying a discount leads to a significant increase in the median number of items purchased (approx. 40% increase). This suggests that discounts are a highly effective tool for increasing basket size and inventory turnover.

**3. Age vs. Spend Demographics (Scatterplot)**

- **What it shows:** This multi-dimensional scatterplot plots customer `Age` against `Total_Spend`, with data points colored by `Gender` and sized by `Items_Purchased`.
- **Why it is important:** It was used to test the hypothesis that spending behavior is tied to age or gender demographics.
- **Insight:** The plot reveals a uniform "cloud" of data points with no discernible pattern. It debunks the age-based hypothesis, proving that the platform has "universal appeal" where a 25-year-old is just as likely to be a high-spender as a 45-year-old. This directs marketing focus away from age.

## 6. Feature Engineering

**Task 1:**

- **`Avg_Item_Value`:** Calculated as `Total_Spend / Items_Purchased` to determine if customers prefer high-value luxury goods or low-cost, high-volume items.
- **`High_Spender`:** A boolean flag identifying customers in the top quartile (75th percentile) of `Total_Spend`, created to isolate and analyze "power users."
- **`Satisfaction_Numeric`:** An ordinal feature (`1-3`) mapped from the `Satisfaction_Level` category to enable correlation analysis.

**Task 2:**

- **`ComplexityScorer` (Type-to-Token Ratio):** A custom transformer was built to calculate the information density of a document. It computes the ratio of unique words to total words (TTR), where a higher TTR indicates more complex, less repetitive text (like an academic abstract) and a lower TTR suggests simple or spammy content.
- **Dimensionality Truncation (TF-IDF + SVD):** Text was vectorized using `TfidfVectorizer` and then its dimensionality was reduced from 500 to 25 components via `TruncatedSVD`. This mimics Matryoshka Representation Learning (MRL), creating a compact, information-rich feature set suitable for lightweight edge models.

## 7. Machine Learning Pipeline

**Task 2:**

- **Pipeline Architecture:** The solution is a modular `scikit-learn` `Pipeline` that integrates custom transformers and a standard classifier.
- **Preprocessing Steps:**
  1. **`TextDistiller`:** Cleans raw input text.
  2. **`FeatureUnion`:** Executes two feature extraction processes in parallel:
     - **Semantic Features:** A `TfidfVectorizer` followed by `TruncatedSVD` to capture the semantic meaning of the text in a low-dimensional space.
     - **Complexity Features:** The custom `ComplexityScorer` to capture the information density.
- **Model Used:** A `RandomForestClassifier` (50 estimators, max depth of 5) was chosen for its strong performance on tabular data, its robustness, and its high efficiency on CPU-only hardware.
- **Design Rationale:** This pipeline was explicitly designed for efficiency. By using `FeatureUnion`, `TruncatedSVD`, and a lightweight `RandomForestClassifier`, it avoids the high computational overhead of deep learning models, making it ideal for real-time triage on edge devices.

## 8. Model Training & Evaluation

**Task 2:**

- **Train-Test Split Strategy:** The dataset was split into an 80% training set and a 20% hold-out test set to ensure the model's performance was validated on unseen data.
- **Evaluation Metrics Used:**
  - **Precision-Recall AUC (PR-AUC):** This was the primary metric. In a triage system, high precision is critical to ensure no junk documents are mistakenly passed to the expensive RAG pipeline. PR-AUC is more informative than standard ROC-AUC on imbalanced or "filter-focused" tasks.
  - **Classification Report:** Provided detailed precision, recall, and F1-scores for both the "Junk" and "High-Value" classes.
  - **Confusion Matrix:** Used to visualize the classifier's performance in separating the two classes and to identify any false positives or false negatives.
- **Model Performance Results:**
  - **PR-AUC:** ~1.00, indicating a highly robust and reliable model.
  - **Accuracy:** 100% on the test set. The model correctly classified all 203 "Junk" documents and all 197 "High-Value" documents.
- **Interpretation of Results:** The outstanding performance demonstrates that combining simple semantic features (TF-IDF + SVD) with a custom-engineered complexity score provides a powerful and sufficient signal to distinguish high-quality academic text from low-quality SMS spam. This validates the feasibility of a lightweight, non-neural triage system.

## 9. Actionable Insights

1.  **Focus Marketing on Membership, Not Age:** EDA overwhelmingly shows that `Membership_Type` is the strongest predictor of high-value customers, while `Age` is irrelevant. Marketing efforts and budget should be reallocated from broad demographic campaigns to targeted initiatives that encourage upgrades to Gold membership.

2.  **Leverage Discounts for Inventory Management:** Discounts are proven to significantly boost the volume of items purchased per transaction. This strategy should be actively used to manage inventory, clear seasonal stock, and increase customer engagement without cannibalizing revenue, as spend correlation remains high.

3.  **A Lightweight Triage Layer Drastically Cuts RAG Costs:** The ML model from Task 2 proves that a simple, CPU-based classifier can filter ~50% of incoming documents with near-perfect accuracy. Implementing this as a "gatekeeper" in a RAG system can reduce downstream processing latency and costs by nearly half, offering a massive ROI for edge and cloud-based AI systems.

## 10. Real-World Usefulness & Impact

**Task 2 Solution:**

- **Who can use this solution?** Developers and organizations implementing Retrieval-Augmented Generation (RAG) systems, particularly those deploying on edge devices (e.g., IoT, mobile) or operating under tight budget constraints.
- **How it helps decision-making:**
  - **Compute Efficiency:** By filtering ~50% of low-information noise at the "gate," the system dramatically reduces the workload on downstream embedding and LLM stages, freeing up resources.
  - **Energy Savings:** Reducing unnecessary inference cycles directly lowers the power consumption of edge devices, which can extend battery life or reduce thermal throttling.
  - **Cost Reduction:** For cloud-integrated systems, this triage layer prevents the use of expensive API tokens (e.g., OpenAI, Cohere) on irrelevant data, leading to direct financial savings.
- **Possible Real-World Deployment Scenario:** An autonomous vehicle's diagnostic system receives thousands of log messages per minute. A triage model running on the vehicle's edge computer instantly filters out routine, low-priority messages. Only critical, high-information warnings are sent to a more powerful RAG system for detailed analysis, ensuring that compute resources are preserved for urgent tasks.

## 11. Tech Stack

- **Python Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `kagglehub`, `arxiv`, `requests`, `feedparser`.
- **Tools and Frameworks:** Jupyter Notebook, Colab, scikit-learn Pipelines.

## 12. How to Run the Project

1.  **Environment Setup:** Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
2.  **Installation Steps:** Install the required libraries from `requirements.txt` (if provided) or install them manually:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn kagglehub arxiv requests feedparser jupyter
    ```
3.  **How to Run Notebooks:**
    - Launch Jupyter Notebook:
      ```bash
      jupyter notebook
      ```
    - In the Jupyter interface, navigate to the project directory and open either `Task_1_DSC.ipynb` or `Task_2_DSC.ipynb`.
    - Run the cells sequentially to reproduce the analysis and model training.
      _(Note: The notebooks are configured to download their respective datasets automatically.)_
