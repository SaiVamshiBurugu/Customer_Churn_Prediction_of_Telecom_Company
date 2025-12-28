# Customer Churn Prediction of Telecom Company

This project aims to predict customer churn for a telecommunications company using various Machine Learning algorithms. The workflow includes data analysis, preprocessing (including handling class imbalance), model selection, and hyperparameter tuning to identify the best performing model.

## 📂 Project Structure
- `churn_prediction.ipynb`: The main Jupyter Notebook containing the entire end-to-end pipeline.
- `Telco-Customer-Churn.csv`: The dataset used for training and testing.

## 🚀 Workflow
The project follows a structured machine learning pipeline:

1.  **Data Exploration (EDA)**:
    - Analyzed dataset structure and distributions.
    - Handled missing values (specifically in `TotalCharges`).
    - Visualized the class imbalance (Churn vs No-Churn).

2.  **Preprocessing**:
    - **Encoding**: Converted categorical variables using Label Encoding.
    - **Scaling**: Standardized numerical features using `StandardScaler`.
    - **Balancing**: Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to handle the class imbalance in the training data.

3.  **Model Selection**:
    - Trained 7 candidate models to compare performance:
        - Logistic Regression
        - Support Vector Machine (SVC)
        - K-Nearest Neighbors (KNN)
        - Random Forest
        - XGBoost
        - Gradient Boosting
        - AdaBoost
    - Automatically selected the **Top 3** models based on the F1-Score.

4.  **Hyperparameter Tuning**:
    - Performed `GridSearchCV` on the top 3 models to optimize their performance.

5.  **Final Evaluation**:
    - Evaluated the tuned models using Accuracy, ROC-AUC, and F1-Score.
    - Generated Confusion Matrices and Classification Reports.

## 🛠️ Requirements
To run this project, you need the following Python libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

## ⚡ How to Run
1.  Clone the repository:
    ```bash
    git clone https://github.com/SaiVamshiBurugu/Customer_Churn_Prediction_of_Telecom_Company.git
    ```
2.  Navigate to the project directory.
3.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook churn_prediction.ipynb
    ```
4.  Run all cells to execute the pipeline.

## 📊 Results
The notebook automatically identifies the best performing models for this specific dataset. Check the output of the "Model Selection" and "Final Evaluation" sections in the notebook for detailed metrics.

---
*Created by Sai Vamshi Burugu*
