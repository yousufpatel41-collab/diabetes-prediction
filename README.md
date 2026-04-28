# 🩺 Diabetes Prediction Using Machine Learning

> **Python · Scikit-learn · XGBoost · SMOTE · SHAP**  
> End-to-end ML pipeline predicting diabetes from patient health metrics — 92.11% accuracy.

---

## 🎯 Project Overview

Built a diabetes prediction system comparing three ML models (Random Forest, Decision Tree, Gradient Boosted Tree) with class balancing via SMOTE and comprehensive model evaluation.

> *Originally prototyped in KNIME; this repository reimplements the full workflow in Python for reproducibility and GitHub sharing.*

---

## 📊 Dataset

- **Source:** [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Size:** 768 patients × 8 clinical features
- **Target:** `Outcome` — 0 (No Diabetes), 1 (Diabetes) — 34.9% positive class

**Features:**
- Pregnancies, Glucose, BloodPressure, SkinThickness
- Insulin, BMI, DiabetesPedigreeFunction, Age

---

## 🏗️ Project Structure

```
diabetes-prediction/
├── data/
│   └── diabetes.csv                # Pima Indians dataset
├── notebooks/
│   └── 01_Model_Comparison.ipynb
├── src/
│   ├── preprocess.py               # cleaning + SMOTE
│   ├── train_all.py                # train & compare all 3 models
│   └── evaluate.py                 # metrics, confusion matrices, ROC
├── models/
│   ├── gradient_boosted.pkl
│   ├── random_forest.pkl
│   └── decision_tree.pkl
├── outputs/
│   ├── model_comparison.png
│   ├── roc_curves_all.png
│   └── confusion_matrices.png
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/yousufpatel41-collab/diabetes-prediction.git
cd diabetes-prediction
pip install -r requirements.txt

# Download diabetes.csv from Kaggle and place in data/
python src/train_all.py
```

---

## 📈 Results

| Model | Accuracy | ROC-AUC | F1 (Diabetes) |
|-------|----------|---------|---------------|
| **Gradient Boosted Tree** ⭐ | **92.11%** | **0.96** | **0.91** |
| Random Forest | 89.61% | 0.94 | 0.87 |
| Decision Tree | 78.57% | 0.78 | 0.74 |

- **SMOTE** improved minority class (Diabetes) recall significantly
- **Gradient Boosted Tree** best-in-class across all metrics
- Cross-validation with 5-fold stratified split for robust evaluation

---

## 🔬 Key Findings

Top predictors (by feature importance):
1. **Glucose** — strongest single predictor
2. **BMI** — high correlation with diabetes risk
3. **Age** — risk increases significantly after 35
4. **DiabetesPedigreeFunction** — family history signal
5. **Pregnancies** — moderate predictive value

---

## 🛠️ Tech Stack

`Python` `Scikit-learn` `XGBoost` `imbalanced-learn` `SHAP` `Pandas` `Matplotlib` `Seaborn`

---

## 📬 Author

**Yousuf Patel** — [LinkedIn](https://linkedin.com/in/yousuf-patel) · [Email](mailto:yousuf9patel@gmail.com)

---

## 📊 Output Charts

### Model Comparison — Accuracy, AUC, F1
![Model Comparison](outputs/model_comparison.png)

### ROC Curves — All 3 Models
![ROC Curves](outputs/roc_curves_all.png)

### Confusion Matrices — All 3 Models
![Confusion Matrices](outputs/confusion_matrices.png)

### Feature Importance (Gradient Boosted Tree)
![Feature Importance](outputs/feature_importance.png)

### Feature Distributions by Diabetes Status
![Feature Distributions](outputs/feature_distributions.png)
