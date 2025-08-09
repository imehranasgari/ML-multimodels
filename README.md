# Project : COVID Vaccination Data Analysis and Modeling

As a data scientist exploring machine learning applications in public health, this project demonstrates my hands-on experience with data preprocessing, imputation techniques, and regression modeling on real-world COVID-19 vaccination data. It showcases my ability to handle incomplete datasets, evaluate multiple models, and visualize learning curves to assess performance—skills directly applicable to roles involving predictive analytics and data-driven decision-making.

## Problem Statement and Goal of Project

The dataset contains global COVID-19 vaccination records with missing values across key metrics like total vaccinations, daily vaccinations, and per-hundred rates. The goal is to load and prepare the data, impute missing values (via a pre-saved pickled DataFrame), perform exploratory analysis, and build  models to identify the best performer (e.g., for prediction tasks). This includes evaluating models like Polynomial Regression and plotting learning curves to understand model generalization.

## Solution Approach

1. **Data Loading and Imputation**: Load the raw CSV dataset and a pre-imputed DataFrame from a pickle file to handle missing values.
2. **Exploratory Data Analysis**: Display data head, summary statistics (describe), and data types/info for initial insights.
3. **Modeling and Evaluation**: Compare  models (e.g., Linear Regression, Polynomial Regression) using grid search for hyperparameter tuning. Select the best model based on performance and plot its learning curve to visualize training and validation scores over varying training set sizes.
4. **Persistence**: Save the imputed DataFrame back to pickle for reusability.

This approach emphasizes robust data handling and model selection, highlighting my methodical process for building reliable ML pipelines.

## Technologies & Libraries

- **Programming Language**: Python 3.8.20 (via Jupyter Notebook)
- **Core Libraries**:
  - Data Manipulation: pandas, numpy
  - Visualization: matplotlib, seaborn, plotly.express, plotly.graph_objects
  - Machine Learning: scikit-learn (StandardScaler, IterativeImputer, SimpleImputer, KNNImputer, BayesianRidge, LinearRegression, Lasso, Ridge, ElasticNet, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, SVR, PolynomialFeatures, Pipeline, GridSearchCV, cross_val_score, mean_squared_error, r2_score, mean_absolute_error, learning_curve)
  - Other Models: xgboost (XGBRegressor), ExtraTreesRegressor
  - Utilities: pickle, joblib (Parallel, delayed)

These tools were selected to cover data preprocessing, imputation, modeling, and parallel computation, demonstrating my familiarity with industry-standard ML ecosystems.

## Description about Dataset

The dataset is loaded from 'country_vaccinations.csv' and includes 31,240 entries with 15 columns:
- Categorical: country, iso_code, date, vaccines, source_name, source_website
- Numerical: total_vaccinations, people_vaccinated, people_fully_vaccinated, daily_vaccinations_raw, daily_vaccinations, total_vaccinations_per_hundred, people_vaccinated_per_hundred, people_fully_vaccinated_per_hundred, daily_vaccinations_per_million

Key statistics (from df.describe()):
- total_vaccinations: Mean ~1.20e+07, Max ~1.43e+09
- daily_vaccinations: Mean ~1.15e+05, Max ~2.24e+07
- Significant missing values in numerical columns (e.g., ~17,451 non-null for total_vaccinations out of 31,240 rows), addressed via imputation.

Data spans vaccination metrics by country and date, sourced from the World Health Organization.

## Installation & Execution Guide

1. **Prerequisites**: Python 3.8+ with Jupyter Notebook. Install dependencies via:
   ```
   pip install pandas matplotlib numpy seaborn pickle5 plotly scikit-learn xgboost joblib
   ```
   (Note: Some libraries like sklearn.experimental.enable_iterative_imputer require enabling experimental features.)

2. **Execution**:
   - Download the notebook (`Project1.ipynb`) and dataset (`country_vaccinations.csv`).
   - Place the pre-imputed `imputed_df.pkl` in the working directory (or run imputation steps to generate it).
   - Open in Jupyter: `jupyter notebook Project1.ipynb`.
   - Run cells sequentially to load data, explore, model, and plot.

This setup ensures reproducibility in a standard Python environment.

## Key Results / Performance

- Best model identified: Polynomial Regression (via Pipeline with PolynomialFeatures and LinearRegression).
- Learning curve plotted for the best model, showing training and cross-validation scores with 5-fold CV, using all CPU cores for efficiency.
- No explicit final metrics (e.g., RMSE, R²) provided in the notebook, but the process includes evaluation via GridSearchCV and cross_val_score.

This highlights my focus on model optimization and interpretability through visualizations.

## Screenshots / Sample Outputs

- **Data Head**:
  ```
         country iso_code        date  total_vaccinations  people_vaccinated  \
  0  Afghanistan      AFG  2021-02-22                 0.0                0.0   
  1  Afghanistan      AFG  2021-02-23                 NaN                NaN   
  ...
  ```

- **Data Head** (HTML table output):
  ```
       country iso_code        date  total_vaccinations  people_vaccinated  \
  0  Afghanistan      AFG  2021-02-22                 0.0                0.0   
  1  Afghanistan      AFG  2021-02-23                 NaN                NaN   
  2  Afghanistan      AFG  2021-02-24                 NaN                NaN   
  3  Afghanistan      AFG  2021-02-25                 NaN                NaN   
  4  Afghanistan      AFG  2021-02-26                 NaN                NaN   
  ```

- **Learning Curve Plot**: Matplotlib figure (size 1000x600) displaying the learning curve for Polynomial Regression.

- **Pickle Equality Check**: Output "True" confirming the saved and loaded DataFrames match.

- **Summary Statistics (df.describe())**:
  ```
         total_vaccinations  people_vaccinated  ...  daily_vaccinations_per_million
  count        1.745100e+04       1.655400e+04  ...                    30948.000000
  mean         1.200444e+07       5.704551e+06  ...                     3426.365969
  ...
  ```

- **Data Info**:
  ```
  <class 'pandas.core.frame.DataFrame'>
  RangeIndex: 31240 entries, 0 to 31239
  Data columns (total 15 columns):
  ...
  ```

- **Learning Curve Plot**: A matplotlib figure (1000x600) showing training and validation curves for Polynomial Regression.

## 👤 Author

## Mehran Asgari
## **Email:** [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com).
## **GitHub:** [https://github.com/imehranasgari](https://github.com/imehranasgari).

---

## 📄 License

This project is licensed under the MIT License – see the `LICENSE` file for details.

💡 *Some interactive outputs (e.g., plots, widgets) may not display correctly on GitHub. If so, please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*