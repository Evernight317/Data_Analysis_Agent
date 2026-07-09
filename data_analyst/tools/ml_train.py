"""Machine learning training tool for the Data Analysis Agent.

Supports regression, classification, and clustering using scikit-learn
with automatic feature engineering, cross-validation, and model persistence.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    # Regression metrics
    r2_score, mean_squared_error, mean_absolute_error,
    # Classification metrics
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix,
    # Clustering metrics
    silhouette_score, calinski_harabasz_score,
)
import joblib

from data_analyst.config import MODELS_DIR
from data_analyst.tools.data_loader import get_dataframe


# --- Algorithm Registry ---

REGRESSION_ALGORITHMS = {
    "linear": "sklearn.linear_model.LinearRegression",
    "ridge": "sklearn.linear_model.Ridge",
    "lasso": "sklearn.linear_model.Lasso",
    "elasticnet": "sklearn.linear_model.ElasticNet",
    "random_forest": "sklearn.ensemble.RandomForestRegressor",
    "gradient_boosting": "sklearn.ensemble.GradientBoostingRegressor",
    "svr": "sklearn.svm.SVR",
    "knn": "sklearn.neighbors.KNeighborsRegressor",
    "decision_tree": "sklearn.tree.DecisionTreeRegressor",
}

CLASSIFICATION_ALGORITHMS = {
    "logistic": "sklearn.linear_model.LogisticRegression",
    "random_forest": "sklearn.ensemble.RandomForestClassifier",
    "gradient_boosting": "sklearn.ensemble.GradientBoostingClassifier",
    "svc": "sklearn.svm.SVC",
    "knn": "sklearn.neighbors.KNeighborsClassifier",
    "decision_tree": "sklearn.tree.DecisionTreeClassifier",
    "naive_bayes": "sklearn.naive_bayes.GaussianNB",
}

CLUSTERING_ALGORITHMS = {
    "kmeans": "sklearn.cluster.KMeans",
    "dbscan": "sklearn.cluster.DBSCAN",
    "agglomerative": "sklearn.cluster.AgglomerativeClustering",
}

# Default hyperparameter grids for tuning
_HYPERPARAM_GRIDS = {
    "random_forest": {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 10, 20],
    },
    "gradient_boosting": {
        "model__n_estimators": [50, 100],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__max_depth": [3, 5],
    },
    "ridge": {
        "model__alpha": [0.1, 1.0, 10.0],
    },
    "lasso": {
        "model__alpha": [0.01, 0.1, 1.0],
    },
    "logistic": {
        "model__C": [0.1, 1.0, 10.0],
        "model__max_iter": [1000],
    },
    "svc": {
        "model__C": [0.1, 1.0, 10.0],
        "model__kernel": ["rbf", "linear"],
    },
    "knn": {
        "model__n_neighbors": [3, 5, 7, 9],
    },
    "kmeans": {
        "model__n_clusters": [2, 3, 4, 5, 6],
    },
}


def _import_algorithm(class_path: str):
    """Dynamically import an algorithm class from its module path."""
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _build_preprocessor(df: pd.DataFrame, feature_columns: list[str]) -> ColumnTransformer:
    """Build a ColumnTransformer for feature preprocessing."""
    numeric_cols = [c for c in feature_columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_columns if c not in numeric_cols]

    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers)


def train_model(
    data_id: str,
    task_type: str,
    target_column: str | None = None,
    feature_columns: list[str] | None = None,
    algorithm: str | None = None,
    tune_hyperparams: bool = False,
    cv_folds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a machine learning model on loaded data.

    Args:
        data_id: Reference to previously loaded data.
        task_type: Type of ML task: "regression", "classification", or "clustering".
        target_column: Target variable column (not needed for clustering).
        feature_columns: Feature columns. If None, uses all columns except target.
        algorithm: Algorithm name. If None, auto-selects based on task_type.
        tune_hyperparams: Whether to perform hyperparameter search.
        cv_folds: Number of cross-validation folds.
        test_size: Test set proportion (0-1).
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with model info, metrics, and file path.
    """
    try:
        df = get_dataframe(data_id)
    except KeyError as e:
        return {"error": str(e)}

    # Validate task type
    if task_type not in ("regression", "classification", "clustering"):
        return {
            "error": f"Invalid task_type: '{task_type}'. "
                     f"Must be 'regression', 'classification', or 'clustering'."
        }

    # Validate target column (not for clustering)
    if task_type != "clustering":
        if not target_column:
            return {"error": f"target_column is required for {task_type} tasks."}
        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found. Available: {list(df.columns)}"}

    # Determine feature columns
    if feature_columns is None:
        if task_type == "clustering":
            feature_columns = df.select_dtypes(include="number").columns.tolist()
        else:
            feature_columns = [c for c in df.columns if c != target_column]
    else:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            return {"error": f"Feature columns not found: {missing}"}

    if not feature_columns:
        return {"error": "No feature columns available."}

    # Auto-select algorithm if not specified
    if algorithm is None:
        if task_type == "regression":
            algorithm = "random_forest"
        elif task_type == "classification":
            algorithm = "random_forest"
        elif task_type == "clustering":
            algorithm = "kmeans"

    # Validate algorithm
    algo_registry = {
        "regression": REGRESSION_ALGORITHMS,
        "classification": CLASSIFICATION_ALGORITHMS,
        "clustering": CLUSTERING_ALGORITHMS,
    }
    available = algo_registry[task_type]
    if algorithm not in available:
        return {"error": f"Algorithm '{algorithm}' not available for {task_type}. Available: {list(available.keys())}"}

    try:
        # Prepare data
        X = df[feature_columns].copy()

        if task_type == "clustering":
            y = None
        else:
            y = df[target_column].copy()
            # Encode categorical target for classification
            if task_type == "classification" and pd.api.types.is_object_dtype(y):
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))

        # Build preprocessing pipeline
        preprocessor = _build_preprocessor(df, feature_columns)

        # Import and instantiate algorithm
        algo_class = _import_algorithm(available[algorithm])
        if algorithm == "kmeans" and task_type == "clustering":
            # Default n_clusters for kmeans
            model = algo_class(n_clusters=3, random_state=random_state, n_init=10)
        elif algorithm in ("dbscan", "agglomerative", "naive_bayes"):
            model = algo_class()
        elif algorithm == "svc":
            model = algo_class(probability=True, random_state=random_state)
        elif algorithm in ("linear", "lasso", "ridge", "elasticnet"):
            # These sklearn models accept random_state only conditionally
            try:
                model = algo_class(random_state=random_state)
            except TypeError:
                model = algo_class()
        elif algorithm == "logistic":
            model = algo_class(max_iter=1000, random_state=random_state)
        else:
            model = algo_class(random_state=random_state)

        # Create full pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        # Training results
        result = {
            "task_type": task_type,
            "algorithm": algorithm,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "data_id": data_id,
        }

        if task_type == "clustering":
            # Fit on all data
            X_processed = preprocessor.fit_transform(X)
            labels = model.fit_predict(X_processed)

            # Clustering metrics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            result["n_clusters"] = n_clusters
            result["cluster_sizes"] = {
                str(i): int(np.sum(labels == i))
                for i in set(labels) if i != -1
            }
            if -1 in labels:
                result["cluster_sizes"]["noise"] = int(np.sum(labels == -1))

            if n_clusters >= 2:
                result["silhouette_score"] = round(float(silhouette_score(X_processed, labels)), 4)
                result["calinski_harabasz_score"] = round(float(calinski_harabasz_score(X_processed, labels)), 4)

            # Hyperparameter tuning for clustering
            if tune_hyperparams and algorithm in _HYPERPARAM_GRIDS:
                grid = {k: v for k, v in _HYPERPARAM_GRIDS[algorithm].items()}
                grid_search = GridSearchCV(
                    pipeline, grid, cv=min(cv_folds, 3),
                    scoring="silhouette" if algorithm == "kmeans" else None,
                    n_jobs=-1,
                )
                grid_search.fit(X)
                pipeline = grid_search.best_estimator_
                result["best_params"] = grid_search.best_params_
                result["best_score"] = round(float(grid_search.best_score_), 4)

        else:
            # Supervised learning: train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=y if task_type == "classification" else None,
            )

            # Hyperparameter tuning
            if tune_hyperparams and algorithm in _HYPERPARAM_GRIDS:
                grid = _HYPERPARAM_GRIDS[algorithm]
                scoring = "r2" if task_type == "regression" else "f1_weighted"
                grid_search = GridSearchCV(
                    pipeline, grid, cv=cv_folds, scoring=scoring, n_jobs=-1,
                )
                grid_search.fit(X_train, y_train)
                pipeline = grid_search.best_estimator_
                result["best_params"] = grid_search.best_params_
                result["best_cv_score"] = round(float(grid_search.best_score_), 4)
            else:
                pipeline.fit(X_train, y_train)

            # Predictions
            y_pred = pipeline.predict(X_test)

            # Metrics
            if task_type == "regression":
                result["metrics"] = {
                    "r2": round(float(r2_score(y_test, y_pred)), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
                    "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
                }
                # Cross-validation scores
                cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring="r2")
                result["cv_r2_mean"] = round(float(cv_scores.mean()), 4)
                result["cv_r2_std"] = round(float(cv_scores.std()), 4)

            elif task_type == "classification":
                result["metrics"] = {
                    "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                    "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
                    "precision_weighted": round(float(precision_score(y_test, y_pred, average="weighted")), 4),
                    "recall_weighted": round(float(recall_score(y_test, y_pred, average="weighted")), 4),
                }
                # AUC if binary
                if len(np.unique(y)) == 2:
                    try:
                        y_proba = pipeline.predict_proba(X_test)[:, 1]
                        result["metrics"]["auc"] = round(float(roc_auc_score(y_test, y_proba)), 4)
                    except (AttributeError, ValueError):
                        pass

                # Classification report
                report = classification_report(y_test, y_pred, output_dict=True)
                result["classification_report"] = report

                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                result["confusion_matrix"] = cm.tolist()

            result["train_size"] = len(X_train)
            result["test_size"] = len(X_test)

        # Save model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_filename = f"{algorithm}_{task_type}_{data_id}.pkl"
        model_path = MODELS_DIR / model_filename
        joblib.dump(pipeline, model_path)

        result["model_path"] = str(model_path)
        result["model_filename"] = model_filename

        return result

    except Exception as e:
        return {"error": f"Model training failed: {str(e)}"}


def get_tool_def() -> dict:
    """Return the Claude tool definition for train_model."""
    return {
        "name": "train_model",
        "description": (
            "Train a machine learning model on loaded data. Supports: "
            "Regression (linear, ridge, lasso, elasticnet, random_forest, gradient_boosting, svr, knn, decision_tree), "
            "Classification (logistic, random_forest, gradient_boosting, svc, knn, decision_tree, naive_bayes), "
            "Clustering (kmeans, dbscan, agglomerative). "
            "Automatically handles feature preprocessing (scaling, encoding, imputation). "
            "Supports cross-validation and hyperparameter tuning. "
            "Saves the trained model for later use with the predict tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "data_id": {
                    "type": "string",
                    "description": "Reference ID of the previously loaded dataset."
                },
                "task_type": {
                    "type": "string",
                    "enum": ["regression", "classification", "clustering"],
                    "description": "Type of ML task."
                },
                "target_column": {
                    "type": "string",
                    "description": "Target variable column name. Required for regression and classification. Not needed for clustering."
                },
                "feature_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Feature column names. If omitted, uses all columns except target (or all numeric for clustering)."
                },
                "algorithm": {
                    "type": "string",
                    "description": "Algorithm to use. If omitted, auto-selects (random_forest for regression/classification, kmeans for clustering)."
                },
                "tune_hyperparams": {
                    "type": "boolean",
                    "description": "Whether to perform hyperparameter search using GridSearchCV. Default: false."
                },
                "cv_folds": {
                    "type": "integer",
                    "description": "Number of cross-validation folds. Default: 5."
                },
                "test_size": {
                    "type": "number",
                    "description": "Proportion of data for testing (0-1). Default: 0.2."
                },
                "random_state": {
                    "type": "integer",
                    "description": "Random seed for reproducibility. Default: 42."
                }
            },
            "required": ["data_id", "task_type"]
        }
    }
