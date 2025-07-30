# Function to calculate a cutoff ratio based on 2 groups

import numpy as np
from sklearn.mixture import GaussianMixture


def calculate_cutoff(df, column):

    # Reshape the data into something we can use for kmeans
    ratios = df[column].to_numpy().reshape(-1, 1)

    # Run a Gaussian Mixture Model to find two different groups of cells
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(ratios)

    # Predict cluster labels
    gmm_labels = gmm.predict(ratios)

    # Access the means of the Gaussian components
    means = gmm.means_.flatten()
    means.sort()  # Sorted so you know which is the low/high cluster

    # Separate the clusters
    group_low = ratios[gmm_labels == np.argmin(means)]
    group_high = ratios[gmm_labels == np.argmax(means)]

    # Find boundary values and cutoff
    lower_max = np.max(group_low)
    upper_min = np.min(group_high)
    cutoff = (lower_max + upper_min) / 2

    return cutoff
