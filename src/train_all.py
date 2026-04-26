"""
train_all.py
Train and compare all three models: Decision Tree, Random Forest, Gradient Boosted Tree.
Applies SMOTE for class balancing. Reproduces KNIME workflow results in Python.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, ConfusionMatrixDisplay, RocCurveDisplay
)
from imblearn.over_sampling import SMOTE

DATA_PATH = "data/diabetes.csv"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ── 1. Load & preprocess ──────────────────────────────────────────────────────

def load_and_clean(filepath: str) -> tuple:
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} patient records")
    print(f"Diabetes prevalence: {df['Outcome'].mean():.1%}")

    # Replace physiologically impossible zeros with median (per-feature)
    zero_impute_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_impute_cols:
        median = df.loc[df[col] != 0, col].median()
        df[col] = df[col].replace(0, median)

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    return X, y


def apply_smote(X_train, y_train):
    print(f"\nBefore SMOTE: {y_train.value_counts().to_dict()}")
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE:  {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


# ── 2. Define models ──────────────────────────────────────────────────────────

MODELS = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosted Tree": GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    ),
}


# ── 3. Train & evaluate ───────────────────────────────────────────────────────

def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    results = {}

    for name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"Training: {name}")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc   = accuracy_score(y_test, y_pred)
        auc   = roc_auc_score(y_test, y_prob)
        f1    = f1_score(y_test, y_pred)

        print(f"  Accuracy: {acc:.4f}")
        print(f"  ROC-AUC:  {auc:.4f}")
        print(f"  F1:       {f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

        results[name] = {
            "model": model,
            "accuracy": acc,
            "roc_auc": auc,
            "f1": f1,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

        # Save model
        safe_name = name.lower().replace(" ", "_")
        with open(f"{MODEL_DIR}/{safe_name}.pkl", "wb") as f:
            pickle.dump(model, f)

    return results


# ── 4. Charts ─────────────────────────────────────────────────────────────────

def chart_model_comparison(results):
    names = list(results.keys())
    metrics = ["accuracy", "roc_auc", "f1"]
    labels  = ["Accuracy", "ROC-AUC", "F1 Score"]
    colors  = ["#3498db", "#2ecc71", "#e74c3c"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric, label, color in zip(axes, metrics, labels, colors):
        vals = [results[n][metric] for n in names]
        bars = ax.bar(names, vals, color=color, edgecolor="white", width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
        ax.set_title(label)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Model Comparison — Diabetes Prediction", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=150)
    plt.close()
    print(f"\n✅ Model comparison chart saved → {OUTPUT_DIR}/model_comparison.png")


def chart_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    for (name, res), color in zip(results.items(), colors):
        RocCurveDisplay.from_predictions(
            y_test, res["y_prob"], ax=ax,
            name=f"{name} (AUC={res['roc_auc']:.3f})",
            color=color,
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC Curves — All Models")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_curves_all.png", dpi=150)
    plt.close()
    print(f"✅ ROC curves saved → {OUTPUT_DIR}/roc_curves_all.png")


def chart_confusion_matrices(results, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, res) in zip(axes, results.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, res["y_pred"],
            display_labels=["No Diabetes", "Diabetes"],
            ax=ax,
            colorbar=False,
        )
        ax.set_title(name)
    fig.suptitle("Confusion Matrices", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrices.png", dpi=150)
    plt.close()
    print(f"✅ Confusion matrices saved → {OUTPUT_DIR}/confusion_matrices.png")


def chart_feature_importance(results, feature_names):
    best = results["Gradient Boosted Tree"]["model"]
    importances = pd.Series(best.feature_importances_, index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax, color="#2c3e50")
    ax.set_title("Feature Importance — Gradient Boosted Tree")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150)
    plt.close()
    print(f"✅ Feature importance saved → {OUTPUT_DIR}/feature_importance.png")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X, y = load_and_clean(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_res, y_res = apply_smote(X_train, y_train)

    results = train_and_evaluate(MODELS, X_res, y_res, X_test, y_test)

    chart_model_comparison(results)
    chart_roc_curves(results, y_test)
    chart_confusion_matrices(results, y_test)
    chart_feature_importance(results, X.columns.tolist())

    # Winner
    best = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\n🏆 Best Model: {best}")
    print(f"   Accuracy: {results[best]['accuracy']:.4f}")
    print(f"   ROC-AUC:  {results[best]['roc_auc']:.4f}")
