
# Churn Prediction – Telecom X (Random Forest + SMOTE)
End‑to‑end churn analysis and prediction for the fictional company Telecom X. The project includes EDA with interactive visuals, data normalization from nested JSON, a robust Machine Learning pipeline (Random Forest) with class balancing via SMOTE and hyperparameter tuning with GridSearchCV, threshold tuning supported by ROC and Precision–Recall curves, model/export utilities, and optional PDF report generation.

Note: This project was developed as part of the Data Science course at Alura. Learn more about Alura here: https://www.alura.com.br/ {target="_blank"}

[Open in Google Colab](https://colab.research.google.com/github/dayanmoshe/Telecom-X---parte-2/blob/main/Modelo_RandomForest_churn_telecom_x.ipynb){target="_blank"}

- Repository: https://github.com/dayanmoshe/Telecom-X---parte-2 {target="_blank"}
- Notebook: Modelo_RandomForest_churn_telecom_x.ipynb

## Overview
The goal is to understand churn drivers and build a predictive model to identify customers likely to cancel their service. The workflow covers:
- Loading and normalizing a semi-structured JSON dataset (customer, phone, internet, account).
- Expanding nested fields and splitting Charges into monthly_value and total_value.
- Exploratory Data Analysis (EDA) with Plotly (interactive charts).
- Preprocessing: categorical encoding (OneHotEncoder), safe numeric conversions.
- Modeling: RandomForestClassifier with class balancing (SMOTE).
- Hyperparameter optimization via GridSearchCV (5‑fold, F1 scoring).
- Model evaluation, feature importance analysis, and an experiment retraining with Top‑10 features.
- Decision threshold calibration with ROC and Precision–Recall analysis.
- Exporting the trained model and preprocessor (Joblib).
- Optional: report generation (PDF) and static image export of charts (Kaleido).

## Dataset and Source
- Data file: TelecomX_Data.json (semi-structured JSON with nested objects)
- Official source: https://raw.githubusercontent.com/alura-cursos/challenge2-data-science/refs/heads/main/TelecomX_Data.json {target="_blank"}

Main columns after normalization:
- Identification: id_client (renamed from customerID), Churn (Yes/No)
- Profile: gender, SeniorCitizen, Partner, Dependents
- Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- Account: Contract, PaperlessBilling, PaymentMethod
- Financial: monthly_value, total_value
- Tenure: tenure (months)

## Key Features
- Visual EDA: churn distribution, churn by contract type, payment method, internet type, phone service, monthly charge, and tenure.
- Data preparation: expand nested JSON to flat columns; Charges split to monthly_value and total_value; numeric coercion.
- Modeling:
  - RandomForestClassifier
  - SMOTE for class balancing
  - GridSearchCV for n_estimators, max_depth, min_samples_split (F1 as the target metric)
- Interpretability: feature_importances_, Top‑10 features selection and retraining trial.
- Threshold tuning: evaluate a range of thresholds (0.1–0.9) to balance precision–recall according to business needs.
- Utilities:
  - Save trained artifacts (model_rf.joblib, preprocessor.joblib)
  - Predict with adjustable threshold and optionally save predictions to CSV
  - Generate a PDF report (FPDF) and export charts to PNG (Kaleido)

## Environment and Installation
Minimum requirements:
- Python 3.10+ (works on 3.12, e.g., in Google Colab)
- pip/venv (or conda)

Setup with pip and venv:
```bash
git clone https://github.com/dayanmoshe/Telecom-X---parte-2.git
cd Telecom-X---parte-2

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

Example requirements.txt (adjust versions to your environment if needed):
```
pandas
requests
numpy
matplotlib
plotly
kaleido
scikit-learn
imbalanced-learn
joblib
fpdf
```

Setup with conda (optional):
```bash
conda create -n telecomx python=3.10 -y
conda activate telecomx
pip install -r requirements.txt
```

## How to Run
Option A — Google Colab (recommended)
- Open the notebook: https://colab.research.google.com/github/dayanmoshe/Telecom-X---parte-2/blob/main/Modelo_RandomForest_churn_telecom_x.ipynb {target="_blank"}
- Run cells in order. The dataset is fetched directly from the public URL.

Option B — Local (Jupyter/VS Code)
- Install dependencies.
- Open Modelo_RandomForest_churn_telecom_x.ipynb.
- Confirm the data URL in the first cells.
- Run all cells.

Expected outputs:
- df_expanded.csv with normalized/expanded data (optional convenience)
- model_rf.joblib and preprocessor.joblib saved to the working directory
- Interactive Plotly charts during EDA
- Validation metrics and reports in the notebook output

## Results and Insights (EDA)
- Contract type:
  - Month‑to‑month plans show the highest churn rate.
  - One‑year and two‑year contracts have lower churn (stronger retention).
- Payment method:
  - “Electronic check” tends to correlate with higher churn compared to more traditional methods.
- Internet service:
  - Fiber‑optic users present higher churn versus DSL or no internet.
- Tenure:
  - Churn is higher in the first months and decreases as tenure grows.
- Monthly charges:
  - Boxplots suggest potential price sensitivity; worth deeper analysis for cost–benefit.

## Modeling and Evaluation
- Classifier: RandomForestClassifier
- Class balancing: SMOTE
- Hyperparameter tuning: GridSearchCV (5 folds, F1 scoring)
- Metrics: accuracy, precision, recall, F1; ROC and Precision–Recall curves
- Suggested decision threshold: 0.4 (good balance to capture churners with solid recall and F1)

Practical recommendation:
- Use threshold = 0.4 for a proactive churn‑prevention strategy.
- Monitor a “gray zone” (e.g., probabilities between 0.4 and 0.6) for prioritized retention actions.

## Using the Trained Model
After training, the notebook saves:
- preprocessor.joblib
- model_rf.joblib

Example: predict churn for a new customer (with threshold = 0.4)
```python
import pandas as pd
import joblib

# 1) Example input (adjust with your fields)
new_input = pd.DataFrame([{
    'id_client': '9999-TESTE',
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 5,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'Yes',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'monthly_value': 89.9,
    'total_value': 450.3
}])

# 2) Load artifacts
preprocessor = joblib.load('preprocessor.joblib')
model_rf = joblib.load('model_rf.joblib')

# 3) Transform and predict
X_new = preprocessor.transform(new_input)
prob = model_rf.predict_proba(X_new)[0, 1]

# 4) Decision with threshold
threshold = 0.4
pred = int(prob >= threshold)

print(f'Churn Probability: {prob:.2f}')
print('Result:', 'Churn' if pred == 1 else 'Not Churn')
```

## Threshold Tuning and Metrics Utilities
The notebook provides:
- ROC and Precision–Recall curves, with AUC‑ROC and Average Precision.
- A routine to test thresholds (0.1 to 0.9) and compare precision, recall, F1, and accuracy.
- Guidance to choose a threshold aligned with business costs/benefits (e.g., prioritize recall to avoid missing churners).

Notes:
- In churn problems, recall on the positive class (churners) is often prioritized to avoid false negatives.
- Use the threshold metrics routine in the notebook to select your operating point.

## Exporting Charts and PDF Reports
- Static images: export Plotly figures with Kaleido.
- PDF report: example using FPDF to generate a summary with insights and recommendations.

Troubleshooting (Kaleido/Plotly)
If you hit an error like “No module named 'plotly.validators.layout.margin'” during export:
```bash
pip uninstall -y plotly
pip install -U plotly kaleido
```
Then restart the kernel/runtime and try a minimal test:
```python
import plotly.express as px
import plotly.io as pio
fig = px.bar(x=["A","B","C"], y=[1,3,2])
pio.write_image(fig, "test_kaleido.png")
```
On Colab, Kaleido generally works without an external Chrome. Restarting the runtime often fixes environment inconsistencies.

## Suggested Project Structure (scalable)
```
.
├── notebooks/
│   └── Modelo_RandomForest_churn_telecom_x.ipynb
├── models/               # saved .joblib artifacts
├── requirements.txt
└── README.md
```

## Roadmap
- Wrap the full pipeline (preprocessor + model) into a single sklearn Pipeline.
- Model versioning and experiment tracking (e.g., DVC/MLflow).
- Time‑aware validation if temporal data is available.
- Explainability: SHAP for global and local explanations.
- Production monitoring: drift and performance tracking.
- Dockerfile + Makefile for reproducibility.
- Optional demo app: Streamlit/Gradio for interactive scoring.

## Contributing
Contributions are welcome! Feel free to open issues and pull requests:
- Open an Issue describing bugs or enhancement ideas.
- Fork, create a feature branch, and open a PR when ready.

## License
Add a LICENSE file in the repository root (e.g., MIT or Apache‑2.0). Update this section accordingly.

## Acknowledgments
- Alura — Data Science course: https://www.alura.com.br/ {target="_blank"}
- Libraries: Scikit‑learn, Imbalanced‑learn, Plotly, Pandas, NumPy, Matplotlib, Joblib, FPDF, Kaleido
- Dataset: TelecomX_Data.json — https://raw.githubusercontent.com/alura-cursos/challenge2-data-science/refs/heads/main/TelecomX_Data.json {target="_blank"}
