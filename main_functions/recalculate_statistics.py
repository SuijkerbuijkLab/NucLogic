# Function that will re-analyze statistics on phenotype and k-nearest neighbors if needed
# This is useful if you want to change the cutoff for the phenotype based on the whole experiment instead of on a per organoid basis

import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd

from utils.calculate_cutoff import calculate_cutoff
from utils.compute_knn_features import compute_knn_features


# As an input we use data, which should be a dataframe of all the cells from all timepoints of all organoids in the experiment you want to recalculate for
# This dataframe should have the columns "sample", "log_ratio", "z", "y", "x"
def recalculate_statistics(data):

    # Find the column that contains log_ratio using regex
    column_name = data.filter(regex="log_ratio_wt_crc").columns[0]

    # Calculate the cutoff based on all data
    cutoff = calculate_cutoff(data, column=column_name)
    print(f"New cutoff calculated: {cutoff}")

    # Loop over data of each sample and determine if it is WT or CRC based on the new cutoff
    new_data = []
    for _, df in data.groupby("sample"):
        percentage_cutoff = (df[column_name] < (cutoff - 0.2)).sum() / len(df) * 100

        # If there are this many cells above the threshold, its likely a pure WT sample
        if percentage_cutoff < 3:
            df["phenotype"] = "wt"
        # If there are this many cells below the threshold, its likely a pure CRC sample
        elif percentage_cutoff > 99:
            df["phenotype"] = "crc"
        # Otherwise we have a mixed sample, where we use the cutoff to determine WT or CRC per cell
        else:
            df["phenotype"] = np.where(df[column_name] < cutoff, "crc", "wt")

        # Loop over every individual frame to recalculate the knn_features
        new_dfs = []
        for _, frame_df in df.groupby("frame"):
            frame_df = compute_knn_features(frame_df, k=5)
            new_dfs.append(frame_df)
        df = pd.concat(new_dfs, ignore_index=True)

        # Append to new data
        new_data.append(df)

    # Make new data one big dataframe again
    new_data = pd.concat(new_data, ignore_index=True)

    return new_data


def solve_gasussians(m1, s1, m2, s2):
    a = 1.0 / (2.0 * s1**2) - 1.0 / (2.0 * s2**2)
    b = m2 / (s2**2) - m1 / (s1**2)
    c = m1**2 / (2 * s1**2) - m2**2 / (2.0 * s2**2) - np.log(s2 / s1)

    return np.roots([a, b, c])


def calculate_cutoff(df, column):

    # Reshape the data into something we can use for kmeans
    ratios = df[column].to_numpy().reshape(-1, 1)

    # Run a Gaussian Mixture Model to find two different groups of cells
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(ratios)

    # To fill in the formula
    mu1, mu2 = gmm.means_.flatten()
    sigma1, sigma2 = np.sqrt(gmm.covariances_.flatten())

    # Solve quadratic formula, and the highest intersection point is the cutoff value
    solved_val = solve_gasussians(mu1, sigma1, mu2, sigma2)
    cutoff = solved_val[1]

    return cutoff
