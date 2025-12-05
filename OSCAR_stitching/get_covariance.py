def get_covariance(var1, var2):
    """
    Calculates the Pearson correlation coefficient between two vectors.

    Args:
        var1: First vector of data
        var2: Second vector of data

    Returns:
        Pearson correlation coefficient
    """
    import numpy as np

    # Convert to numpy arrays if needed
    var1 = np.array(var1)
    var2 = np.array(var2)

    # Check if standard deviations are zero
    std1 = np.std(var1, ddof=1)  # Use ddof=1 for sample std
    std2 = np.std(var2, ddof=1)

    if std1 * std2 == 0:
        pearson_coefficient = 1.0
    else:
        # Calculate covariance and normalize by standard deviations
        covariance = np.cov(var1, var2, ddof=1)[0, 1]  # ddof=1 for sample covariance
        pearson_coefficient = covariance / (std1 * std2)

    return pearson_coefficient
