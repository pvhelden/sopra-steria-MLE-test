import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, GammaRegressor, PoissonRegressor, TweedieRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.svm import SVR

from transformers import add_people_per_room, add_people_per_household


def clean_data(data_path: str):
    """
    Loads housing data from a CSV, converts columns to numeric,
    drops missing values, and separates features from the target.

    Args:
        data_path (str): File path to the housing CSV.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing
            (X, y) where X is the feature DataFrame and
            y is the target Series (median_house_value).
    """
    # Read CSV into a DataFrame
    housing_data = pd.read_csv(data_path)

    # Numeric columns to convert and clean
    num_columns = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population',
        'households', 'median_income'
    ]
    # Convert to numeric and drop rows with NaNs
    for col in num_columns:
        housing_data[col] = pd.to_numeric(housing_data[col], errors='coerce')
    housing_data = housing_data.dropna()

    # Separate features (X) and target (y)
    X = housing_data.drop(columns=['median_house_value'])
    y = housing_data['median_house_value']

    return X, y


def feature_engineering():
    """
    Constructs a pipeline for adding custom features such as
    'people_per_room' and 'people_per_household'.

    Returns:
        Pipeline: A scikit-learn Pipeline that applies
            add_people_per_room and add_people_per_household.
    """
    feature_engineering_pipeline = Pipeline([
        ('add_people_per_room', FunctionTransformer(add_people_per_room)),
        ('add_people_per_household', FunctionTransformer(add_people_per_household)),
    ])
    return feature_engineering_pipeline


def preprocess(X: pd.DataFrame):
    """
    Builds a ColumnTransformer to preprocess both numeric
    and categorical columns, including scaling and one-hot encoding.

    Args:
        X (pd.DataFrame): The raw features DataFrame. Must contain
            'ocean_proximity' for OneHotEncoding and
            newly added features like 'people_per_room', etc.

    Returns:
        ColumnTransformer: A ColumnTransformer that standardizes numeric
            columns and one-hot-encodes the 'ocean_proximity' column.
    """
    # List of numeric columns, including custom feature columns
    num_columns = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households',
        'median_income', 'people_per_room', 'people_per_household',
    ]

    # ColumnTransformer with OneHot for 'ocean_proximity' + StandardScaler for numeric
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(categories=[X['ocean_proximity'].unique()]), ['ocean_proximity']),
            ('num', StandardScaler(), num_columns),
        ]
    )

    return preprocessor


def choose_best_model(feature_engineering_pipeline: Pipeline, preprocessor, X: pd.DataFrame, y: pd.Series):
    """
    Evaluates multiple candidate regression models (XGB, RandomForest,
    LinearRegression, SVR, MLPRegressor) using cross-validation (MAE),
    and prints the best performer.

    Args:
        feature_engineering_pipeline (Pipeline): Pipeline for custom feature creation.
        preprocessor (ColumnTransformer): Preprocessing pipeline (scaling + encoding).
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.

    Returns:
        None
    """
    candidates = [
        ('XGBoost', xgb.XGBRegressor(random_state=42)),
        ('GammaRegressor', GammaRegressor()),
        ('PoissonRegressor', PoissonRegressor()),
        ('TweedieRegressor', TweedieRegressor()),
        ('LinearRegression', LinearRegression()),
        ('SVR', SVR()),
        ('MLPRegressor', MLPRegressor(random_state=42)),
    ]

    best_score = float('inf')
    best_name = None

    # Test each candidate model via CV
    for name, alg in candidates:
        pipeline = Pipeline(steps=[
            ('feature_engineering_pipeline', feature_engineering_pipeline),
            ('preprocessor', preprocessor),
            ('model', alg),
        ])

        # Cross-validate with negative MAE
        cv_scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error', cv=10)
        mae_scores = -cv_scores
        mean_mae = mae_scores.mean()
        median_mae = np.median(mae_scores)
        std_mae = mae_scores.std()

        print(f"{name} Mean MAE: {mean_mae:.0f} (+/- {std_mae:.0f})")
        print(f"\tMedian MAE: {median_mae:.0f}")

        # Track the best (lowest mean MAE)
        if mean_mae < best_score:
            best_score = mean_mae
            best_name = name

    print(f"\nBest algorithm: {best_name} (MAE={best_score:.0f})")


def hyperparameter_tuning_xgbr(feature_engineering_pipeline: Pipeline, preprocessor, X: pd.DataFrame, y: pd.Series):
    """
    Demonstrates hyperparameter tuning for XGBRegressor using GridSearchCV
    (with MAE as the metric). Splits data into train/test to show final test score.

    Args:
        feature_engineering_pipeline (Pipeline): Pipeline for custom feature creation.
        preprocessor (ColumnTransformer): Preprocessing pipeline (scaling + encoding).
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.

    Returns:
        Dict[str, Any]: The best hyperparameters found by the search,
            prefixed with 'model__'.
    """
    # Build a pipeline
    pipeline = Pipeline(steps=[
        ('feature_engineering_pipeline', feature_engineering_pipeline),
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parameter grid to search
    param_grid = {
        'model__n_estimators': [500],
        'model__max_depth': [7],
        'model__learning_rate': [0.07],
        'model__subsample': [1.0],
        'model__colsample_bytree': [0.7],
    }

    # GridSearchCV for negative MAE
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=10,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_  # pipeline with best params
    best_params = grid_search.best_params_  # e.g. {"model__max_depth":7, ...}
    best_score = grid_search.best_score_  # negative MAE

    print("Best Parameters:", best_params)
    print(f"Best CV Score (MAE): {-best_score:.0f}")

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    print(f"Test MAE: {test_mae:.0f}")

    return best_params


def train(feature_engineering_pipeline: Pipeline, preprocessor,
          X: pd.DataFrame, y: pd.Series, model):
    """
    Trains a model within a pipeline that includes feature
    engineering and preprocessing, then saves the pipeline to disk.

    Args:
        feature_engineering_pipeline (Pipeline): Pipeline for custom feature creation.
        preprocessor (ColumnTransformer): Preprocessing pipeline (scaling + encoding).
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target variable.
        model: An uninitialized regression estimator (e.g., xgb.XGBRegressor()).

    Returns:
        None
    """
    # Build a pipeline
    pipeline = Pipeline(steps=[
        ('feature_engineering_pipeline', feature_engineering_pipeline),
        ('preprocessor', preprocessor),
        ('model', model),
    ])

    # Fit the pipeline on the entire dataset
    pipeline.fit(X, y)

    # Save the trained pipeline to disk
    joblib.dump(pipeline, 'housing_price_model.pkl')


def main():
    """
    Main entry point for data cleaning, feature engineering,
    hyperparameter tuning for XGB, and final model training.
    """
    X, y = clean_data('data/housing.csv')

    feat_pipeline = feature_engineering()
    preproc = preprocess(X)

    choose_best_model(feat_pipeline, preproc, X, y)

    best_params = hyperparameter_tuning_xgbr(feat_pipeline, preproc, X, y)
    # 'best_params' has keys like "model__learning_rate", so we remove "model__"
    best_params_cleaned = {key[7:]: value for key, value in best_params.items()}

    # Train final XGBoost model with the best found parameters
    train(feat_pipeline, preproc, X, y, xgb.XGBRegressor(**best_params_cleaned))


if __name__ == '__main__':
    main()
