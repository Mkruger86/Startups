import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path
from startups import fetch_data

# Fetch the data
data = fetch_data()

# Dropping the State column as per your instructions for a start. Might add it back later if i have the time 
data = data.drop('State', axis=1)

# Bin the target variable profit, since this is the target variable
data['profit_bin'] = pd.qcut(data['Profit'], q=5, labels=False)

# Performing stratified sampling
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data['profit_bin']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

# Dropping the bin column after splitting
strat_train_set = strat_train_set.drop('profit_bin', axis=1)
strat_test_set = strat_test_set.drop('profit_bin', axis=1)

# Sort the training set by the Profit column in ascending order
# This is obviously not necessary, but i did it for clarity later one
# (Test set not included as im not supposed to look:))
strat_train_set = strat_train_set.sort_values(by='Profit', ascending=False)
strat_train_set.index.name = 'id'

# File paths for the CSV files to be saved
train_csv_path = Path('stratified_train_set.csv')
test_csv_path = Path('stratified_test_set.csv')

# Check if the CSV files already exist before creating them
if __name__ == "__main__":
    if not train_csv_path.exists():
        strat_train_set.to_csv(train_csv_path, index=True)
        print(f"Stratified Training Set saved to '{train_csv_path}'")
    else:
        print(f"'{train_csv_path}' already exists. Skipping creation.")

    if not test_csv_path.exists():
        strat_test_set.to_csv(test_csv_path, index=False)
        print(f"Stratified Test Set saved to '{test_csv_path}'")
    else:
        print(f"'{test_csv_path}' already exists. Skipping creation.")