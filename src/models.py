from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols)
    ], remainder="drop")

def binary_estimators():
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", solver="liblinear"
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=20, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=42
        )
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            eval_metric="logloss", n_jobs=-1, random_state=42
        )
    except ImportError:
        print("XGBoost is not installed; skipping XGBoost.")
    return models

def multiclass_estimator():
    return RandomForestClassifier(
        n_estimators=180, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42
    )
