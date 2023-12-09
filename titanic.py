import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

import sys
from IPython.core.ultratb import ColorTB

sys.excepthook = ColorTB()

SEED = 42

###############################################################################
############### 1. Load data
# TODO fix cabin process, it's not part of the pipeline

# Read data then drop unused columns
df = pd.read_csv("https://bit.ly/kaggletrain")
df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

# Drop rows where "Embarked" is null. TODO definition for Embarked?
df.dropna(subset=["Embarked"], inplace=True)

# Cabin field - Recode "T" and "G" to "n"
df["Cabin"] = df["Cabin"].astype(str).str[0]  # Get first letter of cabin
df["Cabin"] = (
    df["Cabin"].replace("T", "n").replace("G", "n")
)

###############################################################################
############### 2. SEPARATE X y #############
# Separate data into X,y (Predictors, Target)
X = df.drop("Survived", axis=1)
y = df["Survived"]

###############################################################################
############## 3. COLUMN TRANSFORMER
ct = make_column_transformer(
    (OneHotEncoder(), ["Pclass", "Sex", "Embarked", "Cabin"]),
    (KNNImputer(n_neighbors=2, add_indicator=True), ["Fare", "Age"]),
    remainder="passthrough",
)

ct.fit(X)

###############################################################################
############### 4. Define LogReg model

lr = LogisticRegression(max_iter=100000, solver="lbfgs")

###############################################################################
############### 5. Define Pipeline

pipe = make_pipeline(ct, lr)

pipe

###############################################################################
############### 6. Define Model Parameters

params = {
    "logisticregression__C": [0.01, 0.1, 1.0],
    "logisticregression__penalty": ["l1", "l2"],
}

###############################################################################
############### 7. Grid Search over pipeline

# Gridsearch runs over 'params'
grid = GridSearchCV(
    pipe,
    params,
    cv=5,
    scoring="accuracy",
)

grid.fit(X, y)

pd.DataFrame(grid.cv_results_).sort_values("rank_test_score").to_csv("x.csv")

grid.scorer_

grid.best_score_

grid.best_params_

grid.best_estimator_

df = df.loc[df.Embarked.notna(), ["Survived", "Pclass", "Sex", "Embarked"]]


X = df.loc[:, ["Pclass"]]
y = df.Survived

logreg = LogisticRegression(solver="lbfgs")

cross_val_score(logreg, X, y, cv=5, scoring="accuracy").mean()

y.value_counts(normalize=True)

X = df.drop("Survived", axis="columns")

column_trans = make_column_transformer(
    (OneHotEncoder(), ["Sex", "Embarked"]), remainder="passthrough"
)

column_trans.fit_transform(X)

pipe = make_pipeline(column_trans, logreg)

pipe

cross_val_score(pipe, X, y, cv=5, scoring="accuracy").mean()

# Scratch

# show all available methods
lr.get_params().keys()

pipe = Pipeline(steps=[("preprocessor", column_trans), ("classifier", lr)])
# cross_val_score(clf, X, y, cv = 5, scoring='accuracy').max()

