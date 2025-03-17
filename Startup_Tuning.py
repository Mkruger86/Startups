from Startups_Cleaning import data_Clean, data_Clean_NoAdmin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from pathlib import Path
import numpy as np
import pandas as pd

# Function for grid search on LinearRegression
def grid_search_linear_regression(features, target, description):
    # Define the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    # Parameter grid defining
    param_grid = {
        'scaler': [StandardScaler(), MinMaxScaler()],
        'regressor__fit_intercept': [True, False]
    }
    
    # Initializing GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    
    # Fit the grid search
    grid_search.fit(features, target)
    
    # Get the best parameters and score
    best_params = grid_search.best_params_
    best_score = np.sqrt(-grid_search.best_score_)
    
    print(f"\nBest parameters for {description}: {best_params}")
    print(f"Best RMSE for {description}: {best_score}")
    
    return grid_search.best_estimator_

# Load the test data
test_data = pd.read_csv('stratified_test_set.csv')
features_test = test_data.drop(columns=['Profit'])
target_test = test_data['Profit']

features_Clean = data_Clean.drop(columns=['Profit'])
target_Clean = data_Clean['Profit']

# features_NoAdmin = data_Clean_NoAdmin.drop(columns=['Profit'])
# target_NoAdmin = data_Clean_NoAdmin['Profit']

if __name__ == "__main__":
    # Perform grid search and get the best model
    best_model_Clean = grid_search_linear_regression(features_Clean, target_Clean, "Original Training set")
    # best_model_NoAdmin = grid_search_linear_regression(features_NoAdmin, target_NoAdmin, "Training Set without Administration")
    features_test_Clean = features_test[features_Clean.columns]
    # features_test_NoAdmin = features_test.drop(columns=['Administration'])[features_NoAdmin.columns]
    
    # Evaluate the best model on the test set
    predictions_Clean = best_model_Clean.predict(features_test_Clean)
    rmse_Clean = np.sqrt(mean_squared_error(target_test, predictions_Clean))
    print(f"\nTest RMSE for Original Training set: {rmse_Clean}")
    
       # Save the best prediction to csv
    file_path = Path('best_predictions.csv')
    if not file_path.exists():
        predictions_train_Clean = best_model_Clean.predict(features_Clean)
        predictions_df = pd.DataFrame(predictions_train_Clean, columns=['Predicted_Profit'])
        predictions_df = predictions_df.sort_values(by='Predicted_Profit', ascending=False)  # Sort in descending order
        predictions_df.to_csv(file_path, index=False)
        print(f"Predictions saved to {file_path}")
    else:
        print(f"File {file_path} already exists. Predictions not saved.")

    # predictions_NoAdmin = best_model_NoAdmin.predict(features_test_NoAdmin)
    # rmse_NoAdmin = np.sqrt(mean_squared_error(target_test, predictions_NoAdmin))
    # print(f"\nTest RMSE for Training Set without Administration: {rmse_NoAdmin}")