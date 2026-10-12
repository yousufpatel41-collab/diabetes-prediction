"""Compare three models on the pinned OpenML 37 dataset."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from evaluation import ROOT, compare

DATA_PATH=ROOT/"data/diabetes.csv"

def load_data(path=DATA_PATH):
    df=pd.read_csv(path)
    y=df.pop("Outcome")
    if not y.isin([0,1]).all(): raise ValueError("Outcome must contain 0 and 1")
    # No statistics are fitted here. These zeros become missing values.
    cols=["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    df[cols]=df[cols].replace(0,np.nan)
    return df,y.astype(int)

def build_model(classifier):
    return Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),
        ("resample",SMOTE(random_state=42)),("classifier",classifier)])

def models():
    return {
        "decision_tree":build_model(DecisionTreeClassifier(max_depth=6,min_samples_split=10,min_samples_leaf=5,random_state=42)),
        "random_forest":build_model(RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_split=5,random_state=42,n_jobs=2)),
        "gradient_boosting":build_model(GradientBoostingClassifier(n_estimators=300,max_depth=4,learning_rate=.05,subsample=.8,random_state=42)),
    }

if __name__=="__main__":
    X,y=load_data()
    compare(models(),X,y,DATA_PATH)
