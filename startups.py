import pandas as pd

# Function to fetch the data
def fetch_data():
    datafile = "https://raw.githubusercontent.com/jpandersen61/Machine-Learning/refs/heads/main/50_Startups.csv"
    return pd.read_csv(datafile)

def fetch_train_set():
    trainData = "stratified_train_set.csv"
    return pd.read_csv(trainData)

# Size and type of data
if __name__ == "__main__":
    dataCheck = fetch_data()
    print(dataCheck.head())
    dataCheck.info()
    print(dataCheck["State"].value_counts().rename("State Count"))
