# Titanic EDA

This analysis loads the seaborn Titanic dataset once and immediately saves the loaded frame to `titanic.csv`. If the dataset service is unavailable later, the script uses that committed CSV as its offline fallback. The cleaned analysis frame is saved to `titanic_clean.csv`.

## Missing-value decisions

The raw dataset contains missing values in four columns:

- `age`: 19.865% missing. Values are imputed with the median because the rate is between 5% and 30%.
- `embarked`: 0.224% missing. The two affected rows are dropped because the rate is below 5%.
- `deck`: 77.217% missing. The column is dropped because the rate is above 30%, making imputation unreliable. It is also not needed for the core analysis.
- `embark_town`: 0.224% missing. The affected rows are dropped because the rate is below 5%; the column is also redundant with `embarked`.

The boolean flags `adult_male` and `alone`, plus other derived or redundant descriptive fields, are removed from the analysis frame. The correlation matrix intentionally contains exactly `survived`, `pclass`, `age`, `sibsp`, `parch`, and `fare`.

## Multivariate data story

### Age histogram

The age histogram shows that the cleaned passengers cover a wide adult age range, with more observations concentrated around young and middle adulthood. The distribution is not perfectly symmetric, so the mean and standard deviation should be considered alongside the median when describing age.

### Age box plot

The age box plot makes the central age range and extreme observations easier to compare than the histogram alone. Its outlier count identifies passengers beyond the 1.5-IQR fences, but those observations remain valid passengers and are not removed for this exploratory analysis.

### Fare histogram

The fare histogram is concentrated at lower values with a long tail toward expensive tickets. This shape explains why the fare mean is greater than its median and mode and supports describing fare as right-skewed.

### Fare box plot

The fare box plot highlights the same long upper tail and the many high-fare observations outside the IQR fences. These values are useful evidence about unequal ticket prices, so they are reported as outliers rather than discarded.

### Survival by sex and passenger class

Women have substantially higher observed survival rates than men in every passenger class. Within both sexes, survival generally decreases from first class to third class, showing that sex and passenger class work together rather than explaining the outcome independently.

### Age distribution by survival and sex

The age distributions show how survival differs across both sex and outcome groups. The strong separation by sex remains the dominant pattern, while age provides additional context about the passengers within those groups.

### Fare distribution by class and survival

Fare varies strongly across passenger classes, with higher fares concentrated in higher classes. Because fare partly acts as a proxy for class, its relationship with survival should be interpreted together with the class breakdown rather than as an isolated causal effect.

### Family members aboard and survival

Survival changes across the number of siblings or spouses aboard and is further divided by the number of parents or children aboard. The smallest and largest family-size combinations contain fewer observations, so those extreme groups should be interpreted cautiously.

### Correlation heatmap

The two strongest absolute off-diagonal correlations are between `pclass` and `fare` (r = -0.548) and between `sibsp` and `parch` (r = 0.415). The first reflects the relationship between passenger class and ticket fare, while the second reflects related family-size measures rather than independent causes of survival.

## Exploratory standardization check

The script standardizes `age` and `fare` after cleaning using `z = (x - mean) / std` with sample standard deviation. The resulting `standardization_check.csv` confirms approximately zero means and unit standard deviations; these values are used only for EDA and are not passed to a modeling pipeline.
