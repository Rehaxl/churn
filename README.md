Customer Churn Prediction

This is a  Machine Learning project to predict customer churn (whether a customer is likely to leave or stay) using the Telco Customer Churn dataset. The project includes data cleaning, exploratory data analysis (EDA), model training, evaluation, and testing with new inputs.

Preprocessing :

Cleaned and preprocessed raw Telco dataset.

Performed EDA with meaningful visualizations (customer demographics, contract types, churn rates, etc.).

Built multiple models and selected the best one (Random Forest / Logistic Regression / etc.).

Achieved high prediction accuracy on test data.

Supports real-world prediction by testing with new customer data.


⚙️ Tech Stack

Python 3.x

Pandas, NumPy (Data Preprocessing)

Matplotlib, Seaborn (Visualization)

Scikit-learn (Model building & evaluation)

Joblib / Pickle (Model saving & loading)

📊 Exploratory Data Analysis

Key insights from EDA:

Customers with month-to-month contracts have the highest churn rate.

Senior citizens and customers with no internet service show higher churn.

Longer contract duration is strongly linked with lower churn probability.

![EDA Visuls](churnstock\churnbycontracttype.png)

![Churn Model Visualization](churnstock\seniorcitizen.png)


🤖 Model Training

Trained multiple models including Logistic Regression, Random Forest, and XGBoost.

Best model achieved:

=== Logistic Regression Test Performance ===
Accuracy : 1.0000
Precision: 1.0000
Recall   : 1.0000
F1-score : 1.0000
ROC AUC  : 1.0000

Classification Report:
              precision    recall  f1-score   support

           0     1.0000    1.0000    1.0000      1035
           1     1.0000    1.0000    1.0000       374

    accuracy                         1.0000      1409
   macro avg     1.0000    1.0000    1.0000      1409
weighted avg     1.0000    1.0000    1.0000      1409

![Churn Model Visualization](churnstock\logregscoreconfusionmatrix.png)

![Churn Model Visualization](churnstock\logroccurve.png)


🧪 Model Testing

We can test the model with new customer data using the saved pipeline.

Example:

import pandas as pd
import joblib

# Reload dataset
df = pd.read_excel("Telco_customer_churn.xlsx")
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
df["Total Charges"] = df["Total Charges"].fillna(df["Total Charges"].median())

drop_cols = ["CustomerID","Count","Lat Long","Latitude","Longitude","Churn Reason"]
df_model = df.drop(columns=[c for c in drop_cols if c in df.columns])

X = df_model.drop(columns=["Churn Label"])

# Load model
model = joblib.load("logistic_regression_churn_model.joblib")

# Pick 5 random customers (keep ID safe)
sample = df.sample(100, random_state=42)
test_data = sample.drop(columns=[c for c in drop_cols if c in sample.columns and c != "CustomerID"])
if "Churn Label" in test_data.columns:
    test_data = test_data.drop(columns=["Churn Label"])

# Predictions
sample["Predicted_Churn"] = model.predict(test_data)
sample["Churn_Probability"] = model.predict_proba(test_data)[:, 1]
sample_sorted = sample[["CustomerID", "Predicted_Churn", "Churn_Probability"]].sort_values(
    by="Churn_Probability", ascending=False
)

print(sample_sorted)

# o/p
CustomerID  Predicted_Churn  Churn_Probability
233   6513-EECDB                1           0.998622
1345  8020-BWHYL                1           0.998142
185   2189-WWOEW                1           0.997929
1188  4910-AQFFX                1           0.997924
1090  9821-POOTN                1           0.997578
...          ...              ...                ...
2622  7649-SIJJF                0           0.000909
6006  5286-YHCVC                0           0.000726
5194  5329-KRDTM                0           0.000667
6185  5093-FEGLU                0           0.000518
6685  4891-NLUBA                0           0.000395

[100 rows x 3 columns]



EDA Plot log regression


![Churn Model Visualization](churnstock\edalogistic.png)




📝 How to Run

Clone the repo:

git clone https://github.com/Rehaxl/Churn-Analysis-and-Prediction-Model.git
cd path(run)

Run notebooks for EDA & training.

Test with new customer data using app.py.

📌 Future Improvements

Deploy the model as a Flask / FastAPI web app.

Build an interactive dashboard (Streamlit / Power BI).

Enhance feature engineering with external datasets.

# dataset
[Local Dataset File](churnstock\Telco_customer_churn.xlsx)
