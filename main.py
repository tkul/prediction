import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    mean_absolute_error, 
    mean_absolute_percentage_error
)
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification

def load_and_preprocess_data(file_path):
    try:
        data = pd.read_excel(file_path, sheet_name="Data")
        data = data.astype(float)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

    print("Data Information:")
    data.info()
    
    print("\nDescriptive Statistics:")
    print(data.describe())
    
    print("\nMissing Values:")
    print(data.isnull().sum())

    data = data.fillna(data.mean())
   
    corr_matrix = data.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f")
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()

    return data

def preprocess_data(data: pd.DataFrame):
    X = data.drop(columns=["Y"])
    y = data["Y"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, train_size=0.8, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, X, y

def build_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
   
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    #bu değerleri sunumda yorumlamayı unutmayalım
    print("\nModel Performance Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance in Random Forest Model')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    plt.close()
    
    return model, feature_importance

def hyperparameter_tuning(X_train, X_test, y_train, y_test):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    y_pred_best = best_model.predict(X_test)
    
    mse_best = mean_squared_error(y_test, y_pred_best)
    r2_best = r2_score(y_test, y_pred_best)
    
    print("\nBest Model Performance:")
    print(f"Mean Squared Error (MSE): {mse_best:.4f}")
    print(f"R2 Score: {r2_best:.4f}")

    return best_model


def main():
    file_path = "GR10_Prediction.xlsx"
    
    data = load_and_preprocess_data(file_path)
    
    if data is None:
        return
    
    X_train, X_test, y_train, y_test, X, y = preprocess_data(data)
    
    initial_model, feature_importance = build_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    best_model = hyperparameter_tuning(X_train, X_test, y_train, y_test)

    feature_importance.to_csv('feature_importance.csv', index=False)

if __name__ == "__main__":
    main()