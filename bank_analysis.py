"""Data cleaning, analysis and modelling helpers for the Bank Marketing project.

Used by both the notebook (analysis.ipynb) and the Streamlit app (app.py),
so the two always show the same numbers.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "bank-additional.csv"
RANDOM_STATE = 42

CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "day_of_week", "poutcome"]
NUMERIC = ["age", "campaign", "previous", "previously_contacted",
           "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m",
           "nr.employed"]
MONTH_ORDER = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def load_raw(path=DATA_PATH):
    """Read the original semicolon-separated file."""
    return pd.read_csv(path, sep=";")


def missing_report(raw):
    """Count the 'unknown' labels (the dataset's code for missing values) per column."""
    counts = (raw == "unknown").sum()
    counts = counts[counts > 0].sort_values(ascending=False)
    return pd.DataFrame({"missing": counts,
                         "percent": (100 * counts / len(raw)).round(1)})


def clean(raw):
    """Return the cleaned table and a short log of every cleaning step.

    Decisions:
    * job, housing, loan: few missing rows (under 3%), so those rows are dropped.
    * marital, education: few missing values, filled with the most frequent value.
    * default: about 20% unknown. Clients who did not answer are a group of their own,
      so 'unknown' is kept as a separate category instead of being guessed.
    * pdays: 999 means 'never contacted before'; replaced by a yes/no flag.
    * duration: removed, because the call length is only known after the call ends
      (using it would leak the answer into the model).
    """
    df = raw.copy()
    log = []

    n0 = len(df)
    df = df[~df[["job", "housing", "loan"]].eq("unknown").any(axis=1)]
    log.append(f"Dropped {n0 - len(df)} rows with unknown job, housing or loan "
               f"({100 * (n0 - len(df)) / n0:.1f}% of the data).")

    for col in ["marital", "education"]:
        mode = df.loc[df[col] != "unknown", col].mode()[0]
        n = int((df[col] == "unknown").sum())
        df[col] = df[col].replace("unknown", mode)
        log.append(f"Filled {n} unknown '{col}' values with the most frequent value '{mode}'.")

    log.append(f"Kept {int((df['default'] == 'unknown').sum())} unknown 'default' values "
               "as their own category.")

    df["previously_contacted"] = (df["pdays"] != 999).astype(int)
    df = df.drop(columns=["pdays", "duration"])
    log.append("Replaced pdays with a 0/1 flag 'previously_contacted' and removed "
               "'duration' (only known after the call).")

    df["subscribed"] = (df["y"] == "yes").astype(int)
    df = df.drop(columns="y").reset_index(drop=True)
    return df, log


def rate_by(df, col, order=None, min_count=1):
    """Subscription rate (%) and number of clients for each value of a column."""
    out = (df.groupby(col, observed=True)["subscribed"]
             .agg(clients="size", rate="mean"))
    out["rate"] = (100 * out["rate"]).round(1)
    out = out[out["clients"] >= min_count]
    if order is not None:
        out = out.reindex([o for o in order if o in out.index])
    return out


def add_groups(df):
    """Add banded columns used in the charts."""
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], bins=[17, 29, 39, 49, 59, 100],
                             labels=["18-29", "30-39", "40-49", "50-59", "60+"])
    df["contacts_in_campaign"] = pd.cut(df["campaign"], bins=[0, 1, 2, 3, 5, 100],
                                        labels=["1", "2", "3", "4-5", "6+"])
    return df


def features_target(df):
    return df[CATEGORICAL + NUMERIC], df["subscribed"]


def make_models():
    """The three models compared in the project."""
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])
    return {
        "Baseline (always 'no')": Pipeline([("prep", prep),
                                            ("model", DummyClassifier(strategy="most_frequent"))]),
        "KNN (k=6)": Pipeline([("prep", prep),
                               ("model", KNeighborsClassifier(n_neighbors=6))]),
        "Logistic regression (balanced)": Pipeline([
            ("prep", prep),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000))]),
    }


def cv_predictions(model, X, y, folds=5):
    """Out-of-fold probabilities: every client is scored by a model that never saw them."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def scores(y, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision (yes)": precision_score(y, pred, zero_division=0),
        "Recall (yes)": recall_score(y, pred),
        "F1 (yes)": f1_score(y, pred),
        "ROC AUC": roc_auc_score(y, proba),
    }


def compare_models(X, y):
    """Cross-validated scores and out-of-fold probabilities for every model."""
    rows, probas = {}, {}
    for name, model in make_models().items():
        p = cv_predictions(model, X, y)
        probas[name] = p
        rows[name] = scores(y, p)
    return pd.DataFrame(rows).T.round(3), probas


def gains(y, proba, shares=(0.1, 0.2, 0.3, 0.5)):
    """Share of all subscribers found when calling only the top-scored clients."""
    order = np.argsort(-proba)
    y_sorted = np.asarray(y)[order]
    total = y_sorted.sum()
    rows = []
    for s in shares:
        k = int(round(s * len(y_sorted)))
        found = y_sorted[:k].sum()
        rows.append({"Clients called": f"top {int(s * 100)}%",
                     "Calls": k,
                     "Subscribers found": int(found),
                     "Share of all subscribers": round(100 * found / total, 1),
                     "Success rate of calls (%)": round(100 * found / k, 1)})
    return pd.DataFrame(rows)


def gains_curve(y, proba):
    order = np.argsort(-proba)
    y_sorted = np.asarray(y)[order]
    x = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    return x, np.cumsum(y_sorted) / y_sorted.sum()
