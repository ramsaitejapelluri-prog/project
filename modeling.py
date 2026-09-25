"""Train and compare Titanic classification and fare-regression models."""
from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
RANDOM_STATE = 42
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
NUMERIC_FEATURES = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES = ["sex", "embarked"]


def make_preprocessor() -> ColumnTransformer:
    """Build the train-only imputation, encoding, and scaling steps."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def metric_row(
    name: str,
    actual: pd.Series,
    predicted: np.ndarray,
    probability: np.ndarray,
) -> dict[str, object]:
    return {
        "model": name,
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
        "auc": roc_auc_score(actual, probability),
        "confusion_matrix": confusion_matrix(actual, predicted).tolist(),
    }


def split_classifier_data(data: pd.DataFrame):
    """Create the stratified split and summarize the target balance."""
    features = data[FEATURES]
    target = data["survived"]
    counts = target.value_counts().sort_index()
    proportions = target.value_counts(normalize=True).sort_index()
    balance = {
        "not_survived": {
            "count": int(counts.get(0, 0)),
            "proportion": float(proportions.get(0, 0)),
        },
        "survived": {
            "count": int(counts.get(1, 0)),
            "proportion": float(proportions.get(1, 0)),
        },
    }
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    return train_x, test_x, train_y, test_y, balance


def train_classifiers(train_x, test_x, train_y, test_y):
    """Fit the three classifiers and save the ROC/tree visualizations."""
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
        ),
    }
    rows = []
    pipelines = {}
    roc_figure, roc_axis = plt.subplots(figsize=(8, 6))

    for name, estimator in estimators.items():
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", estimator),
        ])
        pipeline.fit(train_x, train_y)
        pipelines[name] = pipeline
        predicted = np.asarray(pipeline.predict(test_x))
        probability = np.asarray(pipeline.predict_proba(test_x)[:, 1])
        false_positive, true_positive, _ = roc_curve(test_y, probability)
        roc_axis.plot(
            false_positive,
            true_positive,
            label=f"{name} (AUC={roc_auc_score(test_y, probability):.3f})",
        )
        rows.append(metric_row(name, test_y, predicted, probability))

        if name == "Decision Tree":
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            tree_figure, tree_axis = plt.subplots(figsize=(18, 10))
            plot_tree(
                estimator,
                feature_names=feature_names,
                class_names=["not survived", "survived"],
                filled=True,
                rounded=True,
                max_depth=3,
                ax=tree_axis,
            )
            tree_figure.tight_layout()
            tree_figure.savefig(FIGURES / "decision_tree.png", dpi=140)
            plt.close(tree_figure)

    roc_axis.plot([0, 1], [0, 1], "k--")
    roc_axis.set_xlabel("False positive rate")
    roc_axis.set_ylabel("True positive rate")
    roc_axis.set_title("Classifier ROC curves")
    roc_axis.legend()
    roc_figure.tight_layout()
    roc_figure.savefig(FIGURES / "roc_curves.png", dpi=140)
    plt.close(roc_figure)

    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "classification_metrics.csv", index=False)
    return results, pipelines


def compare_imbalance(train_x, test_x, train_y, test_y):
    """Compare no handling, class weights, and train-only SMOTE."""
    strategies = [
        ("baseline", LogisticRegression(max_iter=2000), "passthrough"),
        (
            "class_weight_balanced",
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            "passthrough",
        ),
        (
            "SMOTE_train_only",
            LogisticRegression(max_iter=2000),
            SMOTE(random_state=RANDOM_STATE),
        ),
    ]
    rows = []
    for name, estimator, sampler in strategies:
        pipeline = ImbPipeline([
            ("preprocessor", make_preprocessor()),
            ("sampler", sampler),
            ("model", estimator),
        ])
        pipeline.fit(train_x, train_y)
        predicted = np.asarray(pipeline.predict(test_x))
        rows.append({
            "strategy": name,
            "precision": precision_score(test_y, predicted, zero_division=0),
            "recall": recall_score(test_y, predicted, zero_division=0),
            "f1": f1_score(test_y, predicted, zero_division=0),
        })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(ROOT / "imbalance_comparison.csv", index=False)
    winner = comparison.sort_values(["f1", "recall"], ascending=False).iloc[0]
    conclusion = (
        f"The {winner['strategy']} strategy performed best by F1 "
        f"({winner['f1']:.3f}), with precision={winner['precision']:.3f} "
        f"and recall={winner['recall']:.3f}. It offers the strongest balance "
        "between false positives and false negatives on the held-out test set."
    )
    return comparison, conclusion


def tune_forest(train_x, test_x, train_y, test_y):
    """Tune a Random Forest on training folds and return its test metrics."""
    search_pipeline = Pipeline([
        ("preprocessor", make_preprocessor()),
        (
            "model",
            RandomForestClassifier(
                oob_score=True,
                bootstrap=True,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])
    search = GridSearchCV(
        search_pipeline,
        {
            "model__n_estimators": [150, 300],
            "model__max_depth": [None, 5, 10],
            "model__max_features": ["sqrt", 0.8],
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(train_x, train_y)

    pipeline = search.best_estimator_
    forest = pipeline.named_steps["model"]
    predicted = np.asarray(pipeline.predict(test_x))
    probability = np.asarray(pipeline.predict_proba(test_x)[:, 1])
    metrics = metric_row("Tuned Random Forest", test_y, predicted, probability)
    metrics.update({
        "best_params": search.best_params_,
        "cv_f1": search.best_score_,
        "oob_score": forest.oob_score_,
    })
    return pipeline, metrics


def save_best_pipeline(metrics, pipelines, tuned_pipeline, test_x):
    """Select the strongest classifier candidate and verify its reload."""
    selected = metrics.sort_values(["f1", "auc"], ascending=False).iloc[0]
    name = selected["model"]
    pipeline = tuned_pipeline if name == "Tuned Random Forest" else pipelines[name]
    path = ROOT / "best_titanic_pipeline.joblib"
    joblib.dump(pipeline, path)
    reloaded = joblib.load(path)
    print("Saved deployment pipeline:", name)
    print("Reloaded predictions:", reloaded.predict(test_x.iloc[:5]).tolist())
    return selected, name


def run_fare_regression(data: pd.DataFrame):
    """Fit fare regression, calculate metrics, and save its residual plot."""
    features = data.drop(columns=["fare", "survived"])
    target = data["fare"]
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in features if column not in numeric]
    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical,
        ),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])
    pipeline.fit(train_x, train_y)
    predicted = np.asarray(pipeline.predict(test_x))
    r2 = pipeline.score(test_x, test_y)
    transformed_features = len(preprocessor.get_feature_names_out())
    adjusted_r2 = 1 - (
        (1 - r2) * (len(test_y) - 1)
        / (len(test_y) - transformed_features - 1)
    )
    metrics = {
        "MAE": mean_absolute_error(test_y, predicted),
        "RMSE": mean_squared_error(test_y, predicted) ** 0.5,
        "R2": r2,
        "Adjusted_R2": adjusted_r2,
    }

    residuals = test_y - predicted
    residual_correlation = np.corrcoef(predicted, np.abs(residuals))[0, 1]
    if abs(residual_correlation) >= 0.2:
        conclusion = (
            "The residual plot suggests heteroscedasticity because absolute "
            "residual size changes with predicted fare."
        )
    else:
        conclusion = (
            "The residual plot does not show strong evidence of heteroscedasticity; "
            "the residual spread is reasonably stable across predicted fare."
        )

    figure, axis = plt.subplots(figsize=(7, 4))
    sns.scatterplot(x=predicted, y=residuals, alpha=0.6, ax=axis)
    axis.axhline(0, color="red", linestyle="--")
    axis.set_xlabel("Predicted fare")
    axis.set_ylabel("Residual")
    axis.set_title("Fare regression residuals")
    figure.tight_layout()
    figure.savefig(FIGURES / "fare_residuals.png", dpi=140)
    plt.close(figure)
    pd.DataFrame([metrics]).to_csv(ROOT / "regression_metrics.csv", index=False)
    return metrics, conclusion


def write_outputs(
    classifier_metrics,
    regression_metrics,
    imbalance_metrics,
    balance,
    stratification_note,
    imbalance_conclusion,
    residual_conclusion,
    selected_metrics,
    selected_name,
):
    """Write the combined comparison CSV and human-readable report."""
    comparison = classifier_metrics.copy()
    for column in regression_metrics:
        comparison[column] = np.nan
    comparison = pd.concat([
        comparison,
        pd.DataFrame([{
            "model": "Fare Linear Regression",
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "auc": np.nan,
            "confusion_matrix": np.nan,
            **regression_metrics,
        }]),
    ], ignore_index=True)
    comparison.to_csv(ROOT / "model_comparison.csv", index=False)

    recommendation = (
        f"The recommended deployment classifier is {selected_name}, with "
        f"F1={selected_metrics['f1']:.3f}, AUC={selected_metrics['auc']:.3f}, "
        f"precision={selected_metrics['precision']:.3f}, and "
        f"recall={selected_metrics['recall']:.3f}. It provides the strongest "
        "combined held-out performance and is saved with its complete "
        "preprocessing pipeline. The precision and recall trade-off should "
        "still be reviewed against the operational cost of missed survivors. "
        "Fare regression is a separate task and is not ranked against the "
        "classification metrics."
    )
    report = (
        "# Modeling results\n\n"
        "## Class balance and split\n\n"
        f"{balance}\n\n{stratification_note}\n\n"
        "## Imbalance comparison\n\n"
        f"{imbalance_metrics.to_string(index=False)}\n\n"
        f"{imbalance_conclusion}\n\n"
        "## Regression residuals\n\n"
        f"{residual_conclusion}\n\n"
        "## Model comparison\n\n"
        f"{comparison.to_string(index=False)}\n\n"
        f"{recommendation}\n"
    )
    (ROOT / "modeling_report.md").write_text(report, encoding="utf-8")
    return comparison


def main() -> None:
    data = pd.read_csv(ROOT / "titanic_clean.csv")
    print("Loaded cleaned dataset:", data.shape)
    train_x, test_x, train_y, test_y, balance = split_classifier_data(data)
    stratification_note = (
        "Stratification matters because it preserves the observed class "
        "proportions in both train and test sets."
    )
    print("Class balance before splitting:", balance)
    print(stratification_note)

    classifier_metrics, classifier_pipelines = train_classifiers(
        train_x, test_x, train_y, test_y
    )
    print("\nClassification results:")
    print(classifier_metrics.to_string(index=False))

    imbalance_metrics, imbalance_conclusion = compare_imbalance(
        train_x, test_x, train_y, test_y
    )
    print("\nImbalance comparison:")
    print(imbalance_metrics.to_string(index=False))
    print(imbalance_conclusion)

    tuned_pipeline, tuned_metrics = tune_forest(
        train_x, test_x, train_y, test_y
    )
    print("\nBest Random Forest parameters:", tuned_metrics["best_params"])
    print("Best cross-validation F1:", tuned_metrics["cv_f1"])
    print("Best model OOB score:", tuned_metrics["oob_score"])

    all_candidates = pd.concat([
        classifier_metrics,
        pd.DataFrame([tuned_metrics]),
    ], ignore_index=True)
    selected_metrics, selected_name = save_best_pipeline(
        all_candidates,
        classifier_pipelines,
        tuned_pipeline,
        test_x,
    )
    regression_metrics, residual_conclusion = run_fare_regression(data)
    print("\nFare regression metrics:", regression_metrics)
    print(residual_conclusion)
    write_outputs(
        classifier_metrics,
        regression_metrics,
        imbalance_metrics,
        balance,
        stratification_note,
        imbalance_conclusion,
        residual_conclusion,
        selected_metrics,
        selected_name,
    )


if __name__ == "__main__":
    main()
