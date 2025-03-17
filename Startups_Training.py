from Startups_Cleaning import data_Clean, data_Clean_NoAdmin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Function to train a Linear Regression model and evaluate it on a subset of the training data
def train_model(features, target, description):
    lin_reg_pipeline = make_pipeline(StandardScaler(),LinearRegression())
    lin_reg_pipeline.fit(features, target)
    predictions = lin_reg_pipeline.predict(features)
    
    # Manually selected subset of the training data (indices 20 to 30)
    subset_indices = np.arange(20, 30)
    subset_features = features.iloc[subset_indices]
    subset_target = target.iloc[subset_indices]
    subset_predictions = lin_reg_pipeline.predict(subset_features)
    
    mean_error = np.mean(subset_predictions - subset_target)
    rmse = np.sqrt(mean_squared_error(subset_target, subset_predictions))
    mae = mean_absolute_error(subset_target, subset_predictions)
    
    print(f"\n{description} predictions: {predictions[20:30]}")
    print(f"{description} mean error on subset: {mean_error}")
    print(f"{description} RMSE on subset: {rmse}")
    print(f"{description} MAE on subset: {mae}")

    # Print the coefficients of the trained model
    model = lin_reg_pipeline.named_steps['linearregression']
    coefficients = model.coef_
    feature_names = features.columns
    print(f"{description} model coefficients:")
    for feature, coef in zip(feature_names, coefficients):
        print(f"{feature}: {coef} (absolute: {abs(coef)})")

     # Print the description of the features and target
    print(f"\n{description} features description:")
    print(features.describe())
    print(f"\n{description} target description:")
    print(target.describe())

    # # Cross-validation
    # validation_rmse = cross_val_score(LinearRegression(), features, target, cv=5, scoring='neg_mean_squared_error')
    # clean_rmse = np.sqrt(-validation_rmse)
    # print(f"Cross-Validation RMSE for {description}: {clean_rmse}")
    # print(f"Mean Cross-Validation RMSE for {description}: {np.mean(clean_rmse)}")

# Defining features and target for each dataset
features_Clean = data_Clean.drop(columns=['Profit'])
target_Clean = data_Clean['Profit']

features_NoAdmin = data_Clean_NoAdmin.drop(columns=['Profit'])
target_NoAdmin = data_Clean_NoAdmin['Profit']

if __name__ == "__main__":

    train_model(features_Clean, target_Clean, "Original Training set")
    train_model(features_NoAdmin, target_NoAdmin, "Training Set without Administration")

    # validationRmse = cross_val_score(LinearRegression(), features_Clean, target_Clean, cv=5, scoring='neg_mean_squared_error')
    # cleanRmse = np.sqrt(-validationRmse)
    # print(cleanRmse)
    # print(str(cross_val_score(LinearRegression(), features_NoAdmin, target_NoAdmin, cv=3, scoring='neg_mean_squared_error')))
    # print(str(cross_val_score(LinearRegression(), features_Ratio, target_Ratio, cv=3, scoring='neg_mean_squared_error')))
