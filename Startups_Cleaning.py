import pandas as pd
from startups import fetch_train_set
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Fetch the original training data
data = fetch_train_set()
data_Clean = data.copy()
data_Clean = data_Clean.drop(columns=['id'])

# This training set is used for training a model without administration spend,
## to compare the results of the model with and without,
### as it has the lowest correlation with profit
data_Clean_NoAdmin = data.copy()
data_Clean_NoAdmin = data_Clean_NoAdmin.drop(columns=['id'])
data_Clean_NoAdmin = data_Clean_NoAdmin.drop(columns=['Administration'])

# This training set is used for training a model with administration spend ratio,
## since its apparently more correlated with profit than base administration spend
data_Clean_Ratio = data.copy()
data_Clean_Ratio = data_Clean_Ratio.drop(columns=['id'])
data_Clean_Ratio['Administration_Ratio'] = data_Clean_Ratio['Administration'] / (data_Clean_Ratio['R&D Spend'] + data_Clean_Ratio['Marketing Spend']) * 100
data_Clean_Ratio = data_Clean_Ratio.drop(columns=['Administration'])

# Defining features and target
features = data_Clean.drop(columns=['Profit'])
target = data_Clean['Profit']

# Creating a pipeline
lin_reg_pipeline_nonScaled = make_pipeline(LinearRegression())
lin_reg_pipeline_scaled = make_pipeline(StandardScaler(), LinearRegression())
# Fitting on training data
lin_reg_pipeline_nonScaled.fit(features, target)
lin_reg_pipeline_scaled.fit(features, target)
# Predicting on training data
predictionNonScaled = lin_reg_pipeline_nonScaled.predict(features)
predictionScaled = lin_reg_pipeline_scaled.predict(features)

if __name__ == "__main__":
    print("Non-scaled predictions: ", predictionNonScaled[:10])
    print("Scaled predictions: ", predictionScaled[:10])
    # Calculate and print error metrics
    mae_nonScaled = np.mean(np.abs(target - predictionNonScaled))
    rmse_nonScaled = np.sqrt(mean_squared_error(target, predictionNonScaled))
    mae_scaled = np.mean(np.abs(target - predictionScaled))
    rmse_scaled = np.sqrt(mean_squared_error(target, predictionScaled))

    print("\nError metrics for non-scaled predictions:")
    print(f"MAE: {mae_nonScaled}")
    print(f"RMSE: {rmse_nonScaled}")

    print("\nError metrics for scaled predictions:")
    print(f"MAE: {mae_scaled}")
    print(f"RMSE: {rmse_scaled}")


