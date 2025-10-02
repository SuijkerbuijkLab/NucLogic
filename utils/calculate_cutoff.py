# Function to calculate a cutoff ratio based on 2 groups

import numpy as np
from sklearn.mixture import GaussianMixture


def solve_gasussians(m1, s1, m2, s2):
    a = 1.0 / (2.0 * s1**2) - 1.0 / (2.0 * s2**2)
    b = m2 / (s2**2) - m1 / (s1**2)
    c = m1**2 / (2 * s1**2) - m2**2 / (2.0 * s2**2) - np.log(s2 / s1)
    return np.roots([a, b, c])


def calculate_cutoff(df, column):

    # Make the column names all lower case for easier searching
    column = column.lower()
    df.columns = [col.lower() for col in df.columns]

    # Reshape the data into something we can use for kmeans
    ratios = df[column].to_numpy().reshape(-1, 1)

    # Run a Gaussian Mixture Model to find two different groups of cells
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(ratios)

    # To fill in the formula
    mu1, mu2 = gmm.means_.flatten()
    sigma1, sigma2 = np.sqrt(gmm.covariances_.flatten())

    # Solve quadratic formula, and the highest intersection point is the cutoff value
    if np.std(ratios) > 0.7:  # mixed
        solved_val = solve_gasussians(mu1, sigma1, mu2, sigma2)
        cutoff = solved_val[1]
    elif ratios.mean() > 0:  # wt
        cutoff = ratios.min() - 0.1
    elif ratios.mean() <= 0:  # crc
        cutoff = ratios.max() + 0.1

    return cutoff
