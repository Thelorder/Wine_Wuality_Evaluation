# wine_quality_regression_training.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

import joblib
import os
import warnings
warnings.filterwarnings('ignore')


# --------------------------------------------------
# 1. LOAD & PREPROCESS
# --------------------------------------------------
def load_and_preprocess_data(file_path):
    print("📥 Loading dataset...")
    df = pd.read_csv(file_path)

    if 'type' in df.columns:
        df['type'] = df['type'].replace({'white': 1, 'red': 0})

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())

    print(f"Dataset shape after cleaning: {df.shape}")
    return df


# --------------------------------------------------
# 2. FEATURE / TARGET SPLIT
# --------------------------------------------------
def prepare_features(df):
    print("\n🎯 Preparing features for regression...")

    X = df.drop(['quality'], axis=1)
    y = df['quality']  

    if 'total sulfur dioxide' in X.columns:
        X = X.drop('total sulfur dioxide', axis=1)

    print(f"Features used: {list(X.columns)}")
    return X, y


# --------------------------------------------------
# 3. MODEL TUNING
# --------------------------------------------------
def tune_random_forest_regressor(X_train, y_train):
    print("\n Tuning Random Forest Regressor...")

    param_grid = {
        'n_estimators': [150, 200],
        'max_depth': [12, 18, None],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [3, 5],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )

    grid = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print(f" Best parameters: {grid.best_params_}")
    print(f" Best CV MAE: {-grid.best_score_:.3f}")

    return grid.best_estimator_


# --------------------------------------------------
# 4. EVALUATION
# --------------------------------------------------
def evaluate_model(model, X_train, X_test, y_train, y_test):
    print("\n Model Evaluation")
    print("=" * 40)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    r2 = r2_score(y_test, y_test_pred)

    print(f"Train MAE: {train_mae:.3f}")
    print(f"Test  MAE: {test_mae:.3f}")
    print(f"Train RMSE: {train_rmse:.3f}")
    print(f"Test  RMSE: {test_rmse:.3f}")
    print(f"R² Score: {r2:.3f}")

    return y_test_pred


# --------------------------------------------------
# 5. QUALITY BINS (HUMAN-READABLE OUTPUT)
# --------------------------------------------------
def quality_label(score):
    if score < 5.5:
        return "Low"
    elif score < 6.5:
        return "Standard"
    elif score < 7.5:
        return "High"
    else:
        return "Excellent"


# --------------------------------------------------
# 6. MAIN
# --------------------------------------------------
def main():
    print(" Wine Quality Regression Model")
    print("=" * 50)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "winequalityN.csv")
        
    df = load_and_preprocess_data(data_path)

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = tune_random_forest_regressor(X_train, y_train)

    y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)

    print("\n🔍 Sample Predictions:")
    for i in range(10):
        print(
            f"True: {y_test.iloc[i]} | "
            f"Predicted: {y_test_pred[i]:.2f} | "
            f"Label: {quality_label(y_test_pred[i])}"
        )

    os.makedirs("ml/models", exist_ok=True)
    joblib.dump(model, "ml/models/wine_rf_regressor.pkl")
    joblib.dump(scaler, "ml/models/scaler.pkl")
    joblib.dump(imputer, "ml/models/imputer.pkl")

    print("\n Model saved successfully!")


if __name__ == "__main__":
    main()
