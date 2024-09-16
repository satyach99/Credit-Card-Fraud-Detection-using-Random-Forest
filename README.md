# Credit Card Fraud Detection using Random Forest

## Project Overview

This project aims to build a machine learning model to detect fraudulent credit card transactions. The dataset used contains features related to transactions made by credit card users, with the goal of classifying whether a transaction is fraudulent or not. The model developed utilizes a **Random Forest Classifier** to identify fraudulent transactions with high accuracy.

## Dataset

The dataset is available on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and contains credit card transactions made by European cardholders in September 2013. The dataset is highly imbalanced, with only 0.17% of the transactions classified as fraudulent.

- **Number of rows**: 284,807
- **Number of features**: 31
  - `Time`: Number of seconds elapsed between the transaction and the first transaction in the dataset.
  - `V1-V28`: Result of a PCA transformation to preserve confidentiality.
  - `Amount`: Transaction amount.
  - `Class`: Label for the transaction (`0`: Not Fraudulent, `1`: Fraudulent).

## Project Workflow

1. **Data Loading**: 
   - The dataset is loaded from the CSV file using `pandas`.

2. **Exploratory Data Analysis (EDA)**: 
   - Perform basic statistical analysis to understand the distribution of fraudulent and non-fraudulent transactions.
   - Visualize the data using histograms and correlation matrices to identify patterns.

3. **Data Preprocessing**:
   - Handle missing values (if any).
   - Normalize or scale features where necessary (especially the `Amount` column).

4. **Model Selection**:
   - Implement a **Random Forest Classifier**, which is an ensemble learning method combining multiple decision trees to improve model performance and reduce overfitting.

5. **Model Training & Testing**:
   - Split the dataset into training and testing sets using **train_test_split** from `sklearn`.
   - Train the model using the training set and evaluate it on the test set.

6. **Model Evaluation**:
   - Metrics used for evaluation: **Accuracy**, **Precision**, **Recall**, **F1-score**, and **Confusion Matrix**.
   - Given the imbalanced nature of the dataset, special attention is paid to **Precision** and **Recall** to avoid over-optimizing accuracy at the expense of detecting fraud.

## Installation & Requirements

To run this project, you need the following dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection
   cd credit-card-fraud-detection
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the project directory.

4. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook credit_card_fraud_detection.ipynb
   ```

## Results

- The **Random Forest Classifier** achieved an accuracy of over 99% on the test set. However, given the imbalanced nature of the dataset, more attention was given to **Precision** and **Recall**, which provided better insights into the model's performance in detecting fraudulent transactions.
- **Precision**: ~90%  
- **Recall**: ~85%

## Future Improvements

- Implement **SMOTE** (Synthetic Minority Over-sampling Technique) or other sampling techniques to further address the imbalanced dataset.
- Try other algorithms like **XGBoost** or **Logistic Regression** for comparison.
- Deploy the model using **Flask** or **Streamlit** for real-time fraud detection in a web-based application.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
