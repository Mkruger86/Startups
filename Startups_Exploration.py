import pandas as pd
from startups import fetch_train_set
from pathlib import Path
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# Fetch the training data function
data = fetch_train_set()
# Creating a copy for exploration
data_Explore = data.copy()
# Drop the 'ID' column if it exists
if 'id' in data_Explore.columns:
    data_Explore = data_Explore.drop(columns=['id'])

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

if __name__ == "__main__":
    # Exploring the data
    print (data_Explore.head())
    print (data_Explore.info())
    print (data_Explore.describe())
    corr_matrix = data_Explore.corr(numeric_only=True)
    print (corr_matrix["Profit"].sort_values(ascending=False))

    # Histograms
    data_Explore.hist(bins=10, figsize=(12,8))
    filename_His = Path('trainexplore_histogram.png')
    if not filename_His.exists():
        plt.tight_layout()
        plt.savefig(filename_His)

    # Scatter Matrix
    attributes = ["Profit", "R&D Spend", "Marketing Spend", "Administration"]
    scatter_matrix(data_Explore[attributes], figsize=(12, 8))
    filename_ScM = Path('trainexplore_ScM.png')
    if not filename_ScM.exists():
        plt.tight_layout()
        plt.savefig(filename_ScM)
    
    plt.show()

    # Experimenting with the features (Very basic ones, i know. Will implement proper if time). 
    # Total spend to make ratios for each based on this.
    # Also, the difference between R&D (highest correlation) and Marketing/Administration spend
    # could show importance in impact of one over the other when factoring in R&D spend.
    data_Explore["Total_Spend"] = data_Explore["R&D Spend"] + data_Explore["Marketing Spend"] + data_Explore["Administration"]
    data_Explore["R&D_Spend_Ratio"] = data_Explore["R&D Spend"] / data_Explore["Total_Spend"]
    data_Explore["Marketing_Spend_Ratio"] = data_Explore["Marketing Spend"] / data_Explore["Total_Spend"]
    data_Explore["Admin_Spend_Ratio"] = data_Explore["Administration"] / data_Explore["Total_Spend"]
    data_Explore["R&D_Marketing_Impact"] = data_Explore["R&D Spend"] - data_Explore["Marketing Spend"]
    data_Explore["R&D_Administration_Impact"] = data_Explore["R&D Spend"] - data_Explore["Administration"]
    corr_matrix = data_Explore.corr(numeric_only=True)
    print (corr_matrix["Profit"].sort_values(ascending=False))
