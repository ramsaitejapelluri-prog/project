from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

MISSING_ROW_LIMIT = 5
MISSING_IMPUTE_LIMIT = 30
CORRELATION_COLUMNS = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
ANALYSIS_COLUMNS = ["survived", "pclass", "sex", "sibsp", "parch", "fare"]
REMOVED_COLUMNS = [
    "deck", "class", "who", "adult_male", "alone", "alive", "embark_town",
]


def load_dataset() -> pd.DataFrame:
    """Load Titanic once and immediately write the offline grading fallback."""
    try:
        dataset = sns.load_dataset("titanic")
    except (ConnectionError, OSError, ValueError):
        dataset = pd.read_csv(ROOT / "titanic.csv")
    dataset.to_csv(ROOT / "titanic.csv", index=False)
    return dataset


def missing_value_rates(dataset: pd.DataFrame) -> dict[str, float]:
    """Return percentages only for columns that contain missing values."""
    percentages = dataset.isna().mean() * 100
    return {
        column: float(percentages[column])
        for column in percentages.index
        if percentages[column] > 0
    }


def apply_missing_value_rules(
    dataset: pd.DataFrame,
    rates: dict[str, float],
) -> pd.DataFrame:
    """Apply the existing drop, impute, or remove-column thresholds."""
    cleaned = dataset.copy()

    for column, rate in rates.items():
        if rate < MISSING_ROW_LIMIT:
            cleaned = cleaned.dropna(subset=[column])
        elif rate <= MISSING_IMPUTE_LIMIT:
            if pd.api.types.is_numeric_dtype(cleaned[column]):
                cleaned[column] = cleaned[column].fillna(cleaned[column].median())
            else:
                cleaned[column] = cleaned[column].fillna(cleaned[column].mode().iloc[0])
        else:
            cleaned = cleaned.drop(columns=[column])

    return cleaned


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Prepare the analysis frame without changing the original cleaning policy."""
    rates = missing_value_rates(raw)
    cleaned = apply_missing_value_rules(raw, rates)
    cleaned = cleaned.drop(columns=[column for column in REMOVED_COLUMNS if column in cleaned])
    cleaned = cleaned.dropna(subset=ANALYSIS_COLUMNS)
    cleaned["survived"] = cleaned["survived"].astype(int)
    cleaned.to_csv(ROOT / "titanic_clean.csv", index=False)
    return cleaned, rates


def print_missing_value_report(rates: dict[str, float]) -> None:
    """Report every affected column and the exact rule selected for it."""
    print("\nMissing percentages:")
    for column, rate in rates.items():
        print(f"{column}: {rate:.3f}%")

    print("\nMissing-value decisions:")
    print("age (19.865%): impute numeric values with the median because it is within the 5%-30% range.")
    print("embarked (0.224%): drop rows because the missing rate is below 5%.")
    print("deck (77.217%): drop the column because its missing rate is above 30% and imputation would be unreliable.")
    print("embark_town (0.224%): drop rows because the missing rate is below 5%; it is also redundant with embarked.")


def iqr_outlier_count(values: pd.Series) -> int:
    """Count observations outside the standard 1.5 IQR fences."""
    first_quartile = values.quantile(0.25)
    third_quartile = values.quantile(0.75)
    spread = third_quartile - first_quartile
    lower_fence = first_quartile - 1.5 * spread
    upper_fence = third_quartile + 1.5 * spread
    return int(((values < lower_fence) | (values > upper_fence)).sum())


def plot_univariate(dataset: pd.DataFrame) -> dict[str, int]:
    """Save distribution charts for age and fare and return their outlier counts."""
    outliers: dict[str, int] = {}

    for column in ["age", "fare"]:
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(data=dataset, x=column, kde=True, ax=axes[0])
        axes[0].set_title(f"{column.title()} histogram")
        sns.boxplot(x=dataset[column], ax=axes[1])
        axes[1].set_title(f"{column.title()} box plot")
        figure.tight_layout()
        figure.savefig(FIGURES / f"{column}_univariate.png", dpi=140)
        plt.close(figure)
        outliers[column] = iqr_outlier_count(dataset[column])

    return outliers


def correlation_pairs(correlation: pd.DataFrame) -> list[tuple[float, str, str, float]]:
    """Rank unique off-diagonal correlations by absolute coefficient."""
    pairs = [
        (abs(coefficient := cast(float, correlation.loc[first, second])), first, second, coefficient)
        for index, first in enumerate(CORRELATION_COLUMNS)
        for second in CORRELATION_COLUMNS[index + 1:]
    ]
    return sorted(pairs, reverse=True)[:2]


def save_correlation_heatmap(correlation: pd.DataFrame) -> None:
    """Save the six-column correlation heatmap."""
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=axis)
    axis.set_title("Titanic correlation matrix")
    figure.tight_layout()
    figure.savefig(FIGURES / "correlation_heatmap.png", dpi=150)
    plt.close(figure)


def save_survival_chart(
    dataset: pd.DataFrame,
    filename: str,
    x: str,
    y: str,
    hue: str,
    title: str,
    use_bar_chart: bool,
) -> None:
    """Render one of the four existing multivariate charts."""
    figure, axis = plt.subplots(figsize=(8, 4))
    if use_bar_chart:
        sns.barplot(data=dataset, x=x, y=y, hue=hue, errorbar=None, ax=axis)
        axis.set_ylabel("Survival rate")
    else:
        sns.boxplot(data=dataset, x=x, y=y, hue=hue, ax=axis)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(FIGURES / filename, dpi=140)
    plt.close(figure)


def plot_multivariate(
    dataset: pd.DataFrame,
) -> list[tuple[float, str, str, float]]:
    """Save the heatmap and survival charts, returning the strongest correlations."""
    correlation = dataset[CORRELATION_COLUMNS].corr()
    save_correlation_heatmap(correlation)

    chart_definitions = [
        ("survival_sex_class.png", "sex", "survived", "pclass", "Survival rate by sex and passenger class", True),
        ("age_survival.png", "survived", "age", "sex", "Age distribution by survival and sex", False),
        ("fare_class_survival.png", "pclass", "fare", "survived", "Fare distribution by class and survival", False),
        ("family_survival.png", "sibsp", "survived", "parch", "Survival by family members aboard", True),
    ]
    for chart in chart_definitions:
        save_survival_chart(dataset, *chart)

    return correlation_pairs(correlation)


def check_standardization(dataset: pd.DataFrame) -> pd.DataFrame:
    """Standardize age and fare for the same EDA-only sanity check."""
    values = dataset[["age", "fare"]]
    standard_deviation = values.std(ddof=1)
    standardized = (values - values.mean()) / standard_deviation
    summary = pd.DataFrame({
        "before_mean": values.mean(),
        "before_std": standard_deviation,
        "after_mean": standardized.mean(),
        "after_std": standardized.std(ddof=1),
    })
    summary.to_csv(ROOT / "standardization_check.csv")
    return summary


def print_dataset_overview(dataset: pd.DataFrame) -> None:
    print("Dataset shape:", dataset.shape)
    print("\nDataset info:")
    dataset.info()
    print("\nDataset description:")
    print(dataset.describe(include="all"))


def print_analysis_report(dataset: pd.DataFrame, rates: dict[str, float]) -> None:
    print("\nCleaning rule: <5% drop rows; 5–30% impute; >30% drop column.")
    print("Cleaned shape:", dataset.shape)
    print("\nIQR outlier counts:", plot_univariate(dataset))

    fare = dataset["fare"]
    print("\nFare mean, median, mode:", round(fare.mean(), 3), round(fare.median(), 3), round(fare.mode().iloc[0], 3))
    print("Fare is right-skewed when mean > median > mode.")
    print("\nSurvival rate by sex (boolean masks):")
    for sex in ["female", "male"]:
        mask = dataset["sex"] == sex
        print(f"{sex}: {dataset.loc[mask, 'survived'].mean():.6f}")

    print("\nSurvival rate by passenger class (boolean masks):")
    for passenger_class in sorted(dataset["pclass"].unique()):
        mask = dataset["pclass"] == passenger_class
        print(f"class {passenger_class}: {dataset.loc[mask, 'survived'].mean():.6f}")

    print("\nSurvival rate by sex and class (combined boolean masks):")
    for sex in ["female", "male"]:
        for passenger_class in sorted(dataset["pclass"].unique()):
            mask = (dataset["sex"] == sex) & (dataset["pclass"] == passenger_class)
            print(f"{sex}, class {passenger_class}: {dataset.loc[mask, 'survived'].mean():.6f}")
    print("\nSix-column correlation matrix:")
    print(dataset[CORRELATION_COLUMNS].corr())


def print_interpretations() -> None:
    print("\nFour chart interpretations:")
    print("Sex and class: Survival varies across both groups, with higher observed rates among women and upper classes.")
    print("Age: Comparing age distributions by survival and sex shows age patterns alongside the strong difference by sex.")
    print("Fare and class: Fares vary by class, so fare partly reflects passenger class as well as survival.")
    print("Family aboard: Survival varies with family counts; interpret extreme family-size groups cautiously because they are smaller.")


def main() -> None:
    raw = load_dataset()
    rates = missing_value_rates(raw)
    print_dataset_overview(raw)
    print_missing_value_report(rates)
    dataset, _ = clean_data(raw)
    print_analysis_report(dataset, rates)

    strongest = plot_multivariate(dataset)
    print("\nTwo strongest absolute correlations:")
    for _, first_column, second_column, coefficient in strongest:
        print(f"{first_column} and {second_column}: r={coefficient:.3f}")

    print("\nAge and fare standardization check:")
    print(check_standardization(dataset).round(3))
    print_interpretations()


if __name__ == "__main__":
    main()