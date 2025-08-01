# Function to calculate a cutoff ratio based on 2 groups

import numpy as np
from sklearn.mixture import GaussianMixture


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

    # Predict cluster labels
    gmm_labels = gmm.predict(ratios)

    # Access the means of the Gaussian components
    means = gmm.means_.flatten()
    means.sort()  # Sorted so you know which is the low/high cluster

    # Calculate the stds for the sigmas
    variances = gmm.covariances_.flatten()
    stds = np.sqrt(variances)

    # To fill in the formula
    mu1, mu2 = means
    sigma1, sigma2 = stds

    # Solve quadratic formula, and the highest intersection point is the cutoff value
    solved_val = solve_gasussians(mu1, sigma1, mu2, sigma2)
    cutoff = sorted(solved_val)[-1]

    return cutoff
